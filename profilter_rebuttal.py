import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg
import optax
from profilter import ExtendedKalmanFilter, chol_softplus

class PrOFilter(ExtendedKalmanFilter):
    def __init__(
        self,
        fn_latent,
        fn_obs,
        dynamics_covariance,
        observation_covariance,
        n_inner,
        n_outer,
        *,
        inner_grad_tol=1e-7,
        inner_step_tol=1e-9,
        inner_obj_tol=1e-10,
        outer_mean_tol=1e-6,
        outer_cov_tol=1e-6,
        lbfgs_memory_size=10,
    ):
        super().__init__(
            fn_latent,
            fn_obs,
            dynamics_covariance,
            observation_covariance,
        )

        self.n_inner = n_inner
        self.n_outer = n_outer

        self.inner_grad_tol = inner_grad_tol
        self.inner_step_tol = inner_step_tol
        self.inner_obj_tol = inner_obj_tol

        self.outer_mean_tol = outer_mean_tol
        self.outer_cov_tol = outer_cov_tol

        self.lbfgs_memory_size = lbfgs_memory_size

    def step(self, bel, y, x, callback_fn):
        # Predictive belief:
        #
        #   p(z_t | y_{1:t-1}) = N(a, A)
        #
        bel_pred = super()._predict(bel)

        a = bel_pred.mean
        A = bel_pred.cov
        L_pred = jnp.linalg.cholesky(A)

        R = self.observation_covariance
        state_dim = A.shape[0]

        # chol_softplus only uses the lower-triangular part of its input.
        active_mask = jnp.tril(jnp.ones_like(A))
        n_active = jnp.sum(active_mask)

        def active_rms(M):
            """RMS norm over the active lower-triangular parameters."""
            return jnp.sqrt(
                jnp.sum(jnp.square(M * active_mask)) / n_active
            )

        def mean_change(m, m_old):
            """
            Change in the posterior mean, whitened using the predictive
            covariance A.
            """
            delta = m - m_old

            delta_white = jsp_linalg.solve_triangular(
                L_pred,
                delta,
                lower=True,
            )

            return (
                jnp.linalg.norm(delta_white)
                / jnp.sqrt(state_dim)
            )

        def covariance_change(U, U_old):
            """
            Relative covariance change in predictive-whitened coordinates.

            If

                P = L_pred B B.T L_pred.T,

            then the predictive-whitened covariance is B B.T.
            """
            B = chol_softplus(U)
            B_old = chol_softplus(U_old)

            C = B @ B.T
            C_old = B_old @ B_old.T

            numerator = jnp.linalg.norm(
                C - C_old,
                ord="fro",
            )

            denominator = (
                jnp.sqrt(state_dim)
                + jnp.linalg.norm(C_old, ord="fro")
            )

            return numerator / denominator

        def gen_obj_func(linearisation_mean):
            """
            Construct the profiled objective for a fixed observation
            linearisation point.
            """
            m_lin = linearisation_mean

            H = self.jac_obs(m_lin, x)
            y_hat = self.vobs_fn(m_lin, x)

            HAHt = H @ A @ H.T

            # Residual at the predictive mean under the local linear model:
            #
            #   h(z) ≈ h(m_lin) + H (z - m_lin)
            #
            predictive_error = (
                y
                - y_hat
                - H @ (a - m_lin)
            )

            def obj_func(U):
                # Candidate covariance:
                #
                #   P = L_pred B B.T L_pred.T
                #
                # where B is constrained to be lower triangular with
                # positive diagonal.
                B = chol_softplus(U)
                L_candidate = L_pred @ B
                P = L_candidate @ L_candidate.T

                # Predictive observation covariance under q.
                S = H @ P @ H.T + R

                # Profiled optimum over the candidate posterior mean.
                mean_delta = (
                    A
                    @ H.T
                    @ jnp.linalg.solve(
                        S + HAHt,
                        predictive_error,
                    )
                )

                mean_new = a + mean_delta

                # Observation residual evaluated at the profiled mean.
                residual_y = (
                    y
                    - y_hat
                    - H @ (mean_new - m_lin)
                )

                chol_S = jnp.linalg.cholesky(S)

                whitened_residual_y = jsp_linalg.solve_triangular(
                    chol_S,
                    residual_y,
                    lower=True,
                )

                predictive_quadratic = (
                    0.5
                    * jnp.sum(jnp.square(whitened_residual_y))
                )

                # The U-dependent part of
                #
                #   0.5 log |S| - 0.5 log |P|.
                #
                # Since
                #
                #   |P| = |L_pred|^2 |B|^2,
                #
                # the contribution from L_pred is constant within the
                # current filtering step and can be omitted.
                determinant_term = (
                    jnp.sum(
                        jnp.log(jnp.diagonal(chol_S))
                    )
                    - jnp.sum(
                        jnp.log(jnp.diagonal(B))
                    )
                )

                # Mean component of KL(q || p_pred).
                residual_mean = mean_new - a

                whitened_residual_mean = (
                    jsp_linalg.solve_triangular(
                        L_pred,
                        residual_mean,
                        lower=True,
                    )
                )

                mean_kl = (
                    0.5
                    * jnp.sum(
                        jnp.square(whitened_residual_mean)
                    )
                )

                # Covariance trace component:
                #
                #   tr(A^{-1} P) = tr(B B.T) = ||B||_F^2.
                #
                # Constants such as -state_dim are omitted.
                covariance_kl = (
                    0.5 * jnp.sum(jnp.square(B))
                )

                value = (
                    predictive_quadratic
                    + determinant_term
                    + mean_kl
                    + covariance_kl
                )

                return value, mean_new

            return obj_func

        def inner_loop(outer_carry):
            """
            Solve the covariance problem for one fixed outer
            linearisation point.
            """
            (
                m,
                U,
                m_old,
                U_old,
                outer_count,
            ) = outer_carry

            obj_func = gen_obj_func(m)

            def value_fn(U0):
                value, _ = obj_func(U0)
                return value

            value_and_grad_fn = jax.value_and_grad(
                obj_func,
                has_aux=True,
            )

            solver = optax.lbfgs(
                memory_size=self.lbfgs_memory_size,
                scale_init_precond=True,
            )

            solver_state = solver.init(U)

            (value_init, mean_init), grad_init = (
                value_and_grad_fn(U)
            )

            # Carry:
            #
            #   current U
            #   corresponding profiled mean
            #   L-BFGS state
            #   iteration count
            #   current objective
            #   current gradient
            #   previous U
            #   previous objective
            inner_init = (
                U,
                mean_init,
                solver_state,
                jnp.asarray(0),
                value_init,
                grad_init,
                U,
                value_init,
            )

            def inner_cond(inner_carry):
                (
                    U_current,
                    mean_current,
                    state_current,
                    count,
                    value_current,
                    grad_current,
                    U_previous,
                    value_previous,
                ) = inner_carry

                grad_norm = active_rms(grad_current)

                relative_step = (
                    active_rms(U_current - U_previous)
                    / (
                        1.0
                        + active_rms(U_previous)
                    )
                )

                relative_obj_change = (
                    jnp.abs(
                        value_current - value_previous
                    )
                    / (
                        1.0
                        + jnp.abs(value_previous)
                    )
                )

                gradient_converged = (
                    grad_norm <= self.inner_grad_tol
                )

                # Only test stagnation after at least one update.
                stagnated = (
                    (count > 0)
                    & (
                        relative_step
                        <= self.inner_step_tol
                    )
                    & (
                        relative_obj_change
                        <= self.inner_obj_tol
                    )
                )

                converged = (
                    gradient_converged | stagnated
                )

                finite = (
                    jnp.isfinite(value_current)
                    & jnp.isfinite(grad_norm)
                )

                return (
                    (count < self.n_inner)
                    & (~converged)
                    & finite
                )

            def inner_body(inner_carry):
                (
                    U_current,
                    mean_current,
                    state_current,
                    count,
                    value_current,
                    grad_current,
                    U_previous,
                    value_previous,
                ) = inner_carry

                updates, state_new = solver.update(
                    grad_current,
                    state_current,
                    U_current,
                    value=value_current,
                    grad=grad_current,
                    value_fn=value_fn,
                )

                U_new = optax.apply_updates(
                    U_current,
                    updates,
                )

                # Re-evaluate at U_new so that the objective, gradient,
                # profiled mean, and covariance parameter all correspond
                # to the same iterate.
                (value_new, mean_new), grad_new = (
                    value_and_grad_fn(U_new)
                )

                return (
                    U_new,
                    mean_new,
                    state_new,
                    count + 1,
                    value_new,
                    grad_new,
                    U_current,
                    value_current,
                )

            inner_final = jax.lax.while_loop(
                inner_cond,
                inner_body,
                inner_init,
            )

            U_new = inner_final[0]
            mean_new = inner_final[1]

            return (
                mean_new,
                U_new,
                m,
                U,
                outer_count + 1,
            )

        def outer_cond(outer_carry):
            (
                m,
                U,
                m_old,
                U_old,
                count,
            ) = outer_carry

            delta_mean = mean_change(m, m_old)
            delta_cov = covariance_change(U, U_old)

            converged = (
                (delta_mean <= self.outer_mean_tol)
                & (delta_cov <= self.outer_cov_tol)
            )

            finite = (
                jnp.isfinite(delta_mean)
                & jnp.isfinite(delta_cov)
            )

            # At count == 0, m == m_old and U == U_old by construction.
            # Force the first outer iteration rather than treating this
            # artificial equality as convergence.
            first_iteration = count == 0

            return (
                (count < self.n_outer)
                & finite
                & (
                    first_iteration
                    | (~converged)
                )
            )

        # With the current profilter implementation, chol_softplus uses
        # an exponential diagonal transformation, so a zero raw diagonal
        # corresponds approximately to an identity Cholesky factor.
        U_init = jnp.zeros_like(A)
        m_init = a

        outer_init = (
            m_init,
            U_init,
            m_init,
            U_init,
            jnp.asarray(0),
        )

        final_carry = jax.lax.while_loop(
            outer_cond,
            inner_loop,
            outer_init,
        )

        m_new, U_new, _, _, _ = final_carry

        B_new = chol_softplus(U_new)
        L_new = L_pred @ B_new
        cov_new = L_new @ L_new.T

        bel_update = bel_pred.replace(
            mean=m_new,
            cov=cov_new,
        )

        output = callback_fn(
            bel_update,
            bel_pred,
            y,
            x,
        )

        return bel_update, output

from flax import struct

import jax
import jax.numpy as jnp

@struct.dataclass
class AFGaussState:
    mean: jax.Array
    cov: jax.Array
    innovation_covariance: jax.Array


class AdaptiveFadingKalmanFilter(ExtendedKalmanFilter):
    """
    Adaptive-fading extended Kalman filter.

    The predictive covariance is

        P_{k|k-1}
            = lambda_k F_k P_{k-1|k-1} F_k.T + Q_k,

    where lambda_k >= 1 is estimated by matching the empirical
    innovation covariance to the EKF innovation covariance.

    Parameters
    ----------
    beta:
        Exponential smoothing coefficient for the empirical innovation
        covariance:

            C_k = beta C_{k-1} + (1 - beta) e_k e_k.T.

    lambda_max:
        Upper bound on the adaptive fading factor.

    observation_dim:
        Required when observation_covariance is scalar but the observation
        itself is multidimensional. Otherwise inferred from R.

    eps:
        Numerical tolerance used when estimating lambda_k.
    """

    def __init__(
        self,
        fn_latent,
        fn_obs,
        dynamics_covariance,
        observation_covariance,
        beta=0.98,
        lambda_max=10.0,
        observation_dim=None,
        eps=1e-8,
    ):
        super().__init__(
            fn_latent=fn_latent,
            fn_obs=fn_obs,
            dynamics_covariance=dynamics_covariance,
            observation_covariance=observation_covariance,
        )

        '''
        if not 0.0 <= beta < 1.0:
            raise ValueError("beta must satisfy 0 <= beta < 1.")

        if lambda_max < 1.0:
            raise ValueError("lambda_max must satisfy lambda_max >= 1.")

        if eps <= 0.0:
            raise ValueError("eps must be positive.")
        '''
        
        self.beta = beta
        self.lambda_max = lambda_max
        self.eps = eps

        Rt = jnp.asarray(observation_covariance)

        if observation_dim is not None:
            self.observation_dim = int(observation_dim)
        elif Rt.ndim == 0:
            self.observation_dim = 1
        elif Rt.ndim == 1:
            self.observation_dim = Rt.shape[0]
        else:
            self.observation_dim = Rt.shape[-1]

    @staticmethod
    def _as_covariance_matrix(covariance, dim, dtype=None):
        """
        Convert a scalar, diagonal vector, or full covariance into a matrix.
        """
        covariance = jnp.asarray(covariance, dtype=dtype)

        if covariance.ndim == 0:
            return covariance * jnp.eye(dim, dtype=covariance.dtype)

        if covariance.ndim == 1:
            if covariance.shape[0] != dim:
                raise ValueError(
                    "Diagonal covariance length does not match dimension."
                )
            return jnp.diag(covariance)

        if covariance.shape != (dim, dim):
            raise ValueError(
                f"Expected covariance shape {(dim, dim)}, "
                f"received {covariance.shape}."
            )

        return covariance

    def init_bel(
        self,
        mean,
        cov=1.0,
        innovation_covariance=None,
    ):
        """
        Initialise the AFKF belief.

        The parent method initialises the vectorised state representation and
        Jacobian functions. The resulting Gaussian state is then augmented
        with the running innovation-covariance estimate.
        """
        base_bel = super().init_bel(mean, cov)

        if innovation_covariance is None:
            innovation_covariance = self._as_covariance_matrix(
                self.observation_covariance,
                self.observation_dim,
                dtype=base_bel.mean.dtype,
            )
        else:
            innovation_covariance = self._as_covariance_matrix(
                innovation_covariance,
                self.observation_dim,
                dtype=base_bel.mean.dtype,
            )

        return AFGaussState(
            mean=base_bel.mean,
            cov=base_bel.cov,
            innovation_covariance=innovation_covariance,
        )

    def _predict_mean_and_covariance(self, bel):
        """
        Compute the EKF mean prediction and the unfaded propagated covariance.

        Returns
        -------
        mean_pred:
            Predicted state mean.

        propagated_cov:
            F_k P_{k-1|k-1} F_k.T, before adding Q or applying fading.
        """
        Ft = self.jac_latent(bel.mean)

        mean_pred = self.vlatent_fn(bel.mean)

        propagated_cov = Ft @ bel.cov @ Ft.T
        propagated_cov = 0.5 * (
            propagated_cov + propagated_cov.T
        )

        return mean_pred, propagated_cov

    def _update_innovation_covariance(
        self,
        innovation_covariance,
        innovation,
    ):
        """
        Exponentially weighted innovation-covariance update.
        """
        innovation_outer = jnp.outer(
            innovation,
            innovation,
        )

        covariance_update = (
            self.beta * innovation_covariance
            + (1.0 - self.beta) * innovation_outer
        )

        return 0.5 * (
            covariance_update + covariance_update.T
        )

    def _estimate_fading_factor(
        self,
        propagated_cov,
        innovation_covariance,
        Ht,
        Qt,
        Rt,
    ):
        """
        Estimate the scalar fading factor for vector-valued observations.

        Let

            M_k = H_k A_k H_k.T,

        where

            A_k = F_k P_{k-1|k-1} F_k.T,

        and let

            N_k = C_k - H_k Q_k H_k.T - R_k.

        Under exact covariance matching,

            N_k = lambda_k M_k.

        Therefore,

            M_k^{-1} N_k = lambda_k I,

        and the scalar fading factor is estimated as

            lambda_k
                = (1 / m) trace(M_k^{-1} N_k),

        where m is the observation dimension.

        A linear solve is used instead of explicitly forming M_k^{-1}.
        """
        # M_k = H_k A_k H_k.T
        M = Ht @ propagated_cov @ Ht.T

        # N_k = C_k - H_k Q_k H_k.T - R_k
        fixed_component = Ht @ Qt @ Ht.T + Rt
        N = innovation_covariance - fixed_component

        # Remove minor numerical asymmetry.
        M = 0.5 * (M + M.T)
        N = 0.5 * (N + N.T)

        dim_observation = M.shape[0]
        eye = jnp.eye(dim_observation, dtype=M.dtype)

        # Scale-aware regularization. This is mainly needed when M is singular
        # or nearly singular in one or more observation directions.
        mean_scale = jnp.trace(M) / dim_observation
        jitter = self.eps * jnp.maximum(mean_scale, 1.0)

        M_regularized = M + jitter * eye

        # Solve M X = N, so X = M^{-1} N.
        normalized_mismatch = jnp.linalg.solve(
            M_regularized,
            N,
        )

        lambda_raw = (
            jnp.trace(normalized_mismatch)
            / dim_observation
        )

        lambda_raw = jnp.nan_to_num(
            lambda_raw,
            nan=1.0,
            posinf=self.lambda_max,
            neginf=1.0,
        )

        return jnp.clip(
            lambda_raw,
            min=1.0,
            max=self.lambda_max,
        )

    def _adaptive_predict(self, bel, y, x):
        """
        Perform the adaptive-fading prediction.

        The observation is required because lambda_k is estimated from the
        innovation before constructing the predictive covariance.
        """
        mean_pred, propagated_cov = (
            self._predict_mean_and_covariance(bel)
        )

        yhat = jnp.atleast_1d(
            self.vobs_fn(mean_pred, x)
        )

        y = jnp.atleast_1d(y)

        Ht = jnp.atleast_2d(
            self.jac_obs(mean_pred, x)
        )

        innovation = y - yhat

        dim_latent = mean_pred.shape[0]
        dim_observation = yhat.shape[0]

        Qt = self._as_covariance_matrix(
            self.dynamics_covariance,
            dim_latent,
            dtype=bel.cov.dtype,
        )

        Rt = self._as_covariance_matrix(
            self.observation_covariance,
            dim_observation,
            dtype=bel.cov.dtype,
        )

        innovation_covariance = (
            self._update_innovation_covariance(
                bel.innovation_covariance,
                innovation,
            )
        )

        lambda_k = self._estimate_fading_factor(
            propagated_cov=propagated_cov,
            innovation_covariance=innovation_covariance,
            Ht=Ht,
            Qt=Qt,
            Rt=Rt,
        )

        cov_pred = (
            lambda_k * propagated_cov
            + Qt
        )

        cov_pred = 0.5 * (
            cov_pred + cov_pred.T
        )

        bel_pred = bel.replace(
            mean=mean_pred,
            cov=cov_pred,
            innovation_covariance=innovation_covariance,
        )

        return (
            bel_pred,
            yhat,
            Ht,
            Rt,
            innovation,
            lambda_k,
        )

    def step(self, bel, y, x, callback_fn):
        """
        Run one AFKF prediction-update step.
        """
        (
            bel_pred,
            yhat,
            Ht,
            Rt,
            innovation,
            lambda_k,
        ) = self._adaptive_predict(
            bel,
            y,
            x,
        )

        y = jnp.atleast_1d(y)

        # The inherited update uses bel.replace(mean=..., cov=...).
        # Since bel_pred is an AFGaussState, its innovation covariance
        # is retained automatically.
        bel_update = self._update(
            bel=bel_pred,
            y=y,
            x=x,
            yhat=yhat,
            Ht=Ht,
            Rt=Rt,
        )

        callback_output = callback_fn(
            bel_update,
            bel_pred,
            y,
            x,
        )

        step_output = {
            "callback": callback_output,
            "lambda": lambda_k,
            "innovation": innovation,
            "innovation_covariance": (
                bel_update.innovation_covariance
            ),
        }

        return bel_update, step_output

    def scan(self, bel, y, X, callback_fn=None):
        """
        Run the AFKF over a sequence.

        Returns
        -------
        bel_final:
            Final AFGaussState.

        history:
            Pytree containing:

            - history["callback"]
            - history["lambda"]
            - history["innovation"]
            - history["innovation_covariance"]
        """
        callback_fn = (
            callbacks.get_null
            if callback_fn is None
            else callback_fn
        )

        def _step(bel, yX):
            yt, xt = yX

            return self.step(
                bel,
                yt,
                xt,
                callback_fn,
            )

        bel_final, history = jax.lax.scan(
            _step,
            bel,
            (y, X),
        )

        return bel_final, history