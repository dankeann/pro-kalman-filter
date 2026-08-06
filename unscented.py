import jax
import jax.numpy as jnp
import jaxopt
from profilter import ExtendedKalmanFilter
from profilter import chol_softplus
import optax
import jax.scipy.linalg as jsp_linalg



def unscented_transform(mean, cov, f, alpha=1e-3, beta=2.0, kappa=0.0):
    d = mean.shape[0]
    lam = alpha**2 * (d + kappa) - d

    S = jnp.linalg.cholesky((d + lam) * cov)
    sigma_points = jnp.concatenate([
        mean[None, :],
        mean[None, :] + S.T,
        mean[None, :] - S.T,
    ], axis=0)

    Wm = jnp.concatenate([
        jnp.array([lam / (d + lam)]),
        jnp.full(2 * d, 1.0 / (2 * (d + lam)))
    ])
    Wc = Wm.at[0].add(1 - alpha**2 + beta)

    transformed = jnp.stack([f(sp) for sp in sigma_points])
    mean_pred = jnp.sum(Wm[:, None] * transformed, axis=0)
    diff = transformed - mean_pred
    cov_pred = jnp.einsum('i,ij,ik->jk', Wc, diff, diff)

    return mean_pred, cov_pred


class UnscentedKalmanFilter(ExtendedKalmanFilter): #overriding the predict step in the extended kalman filter
    def _predict(self, bel):
        mean_pred, cov_pred = unscented_transform(bel.mean, bel.cov, self.vlatent_fn)
        cov_pred = cov_pred + self.dynamics_covariance
        bel = bel.replace(mean=mean_pred, cov=cov_pred)
        return bel


# Methods copy and pasted from profilter_rebuttal.py - UKFPrOFilter

class UKFPrOFilter(UnscentedKalmanFilter):
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

            first_iteration = count == 0

            return (
                (count < self.n_outer)
                & finite
                & (
                    first_iteration
                    | (~converged)
                )
            )

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


# Methods Copy and Pasted from profilter.py - IteratedPrOFilter
class IteratedUKFPrOFilter(UnscentedKalmanFilter):
    def __init__(
        self, fn_latent, fn_obs, dynamics_covariance, observation_covariance, n_inner, n_outer,
    ):
        super().__init__(fn_latent, fn_obs, dynamics_covariance, observation_covariance)
        self.n_inner = n_inner
        self.n_outer = n_outer

    def step(self, bel, y, x, callback_fn):
        bel_pred = super()._predict(bel)
        L_pred = jnp.linalg.cholesky(bel_pred.cov)
        Rt = self.observation_covariance

        def update_kwargs(m):
            Ht = self.jac_obs(m, x)
            yhat = self.vobs_fn(m, x) + Ht @ (bel_pred.mean - m)
            At = Ht @ bel_pred.cov @ Ht.T
            H_bar = Ht @ L_pred
            return {"Ht": Ht, "yhat": yhat, "At": At, "H_bar": H_bar}

        U_init = jnp.zeros_like(bel_pred.cov)

        def mean_update(P0, Ht, yhat, At, H_bar):
            S = H_bar @ P0 @ H_bar.T + Rt
            dm = bel_pred.cov @ Ht.T @ jnp.linalg.solve(S + Ht @ bel_pred.cov @ Ht.T, y-yhat)
            return bel_pred.mean + dm

        def mm_obj(U, W_curr, Ht, yhat, At, H_bar):
            L = chol_softplus(U)
            P0 = L @ L.T
            P_diag = jnp.diagonal(P0)
            t1 = jnp.sum(P_diag)
            M = H_bar @ P0 @ H_bar.T + Rt + At
            d = y - yhat
            t2 = jnp.sum(d * jnp.linalg.solve(M, d))
            L_diag = jnp.diagonal(L)
            t3 = -2*jnp.sum(jnp.log(L_diag))
            t4 = jnp.sum(jnp.diagonal(W_curr @ P0))
            return 0.5 * (t1 + t2 + t3 + t4)

        @jax.jit
        def solve_inner_lbfgs(
            U0,
            kwargs,
            *,
            W_curr,
            max_inner_iters=5,
            gtol=1e-4,
            rtol=1e-7,
            patience=3,
        ):
            """
            Runs JAXopt LBFGS inside a lax.while_loop with custom stopping:
              stop if grad_norm <= gtol, OR
                      relative objective improvement <= rtol for `patience` consecutive steps,
                      OR iter reaches max_inner_iters.

            Returns: U_star, diagnostics dict.
            """
            # value_and_grad for JAXopt

            def fun(U):
                return mm_obj(U, W_curr=W_curr, **kwargs)

            vg = jax.value_and_grad(fun)

            solver = jaxopt.LBFGS(fun=fun, value_and_grad=False, has_aux=False, maxiter=1)  # do 1 step per update

            # initialize
            state = solver.init_state(U0)
            f0, g0 = vg(U0)
            gnorm0 = jnp.linalg.norm(g0)

            carry = dict(
                U=U0,
                state=state,
                f=f0,
                gnorm=gnorm0,
                it=jnp.array(0, dtype=jnp.int32),
                stall=jnp.array(0, dtype=jnp.int32),
            )

            def cond(c):
                # continue while not converged and it < max
                not_max = c["it"] < max_inner_iters
                grad_ok = c["gnorm"] > gtol
                stall_ok = c["stall"] < patience
                return not_max & (grad_ok & stall_ok)

            def body(c):
                # one LBFGS update
                step = solver.update(c["U"], c["state"])  # one iteration only
                U1, state1 = step.params, step.state

                f1, g1 = vg(U1)
                gnorm1 = jnp.linalg.norm(g1)

                # relative improvement
                rel_impr = (c["f"] - f1) / jnp.maximum(1.0, jnp.abs(c["f"]))
                stall1 = jnp.where(rel_impr <= rtol, c["stall"] + 1, 0)

                return dict(
                    U=U1,
                    state=state1,
                    f=f1,
                    gnorm=gnorm1,
                    it=c["it"] + 1,
                    stall=stall1,
                )

            out = jax.lax.while_loop(cond, body, carry)

            info = dict(
                iters=out["it"],
                f=out["f"],
                gnorm=out["gnorm"],
                stall=out["stall"],
            )
            return out["U"]

        def body_fun(i, carry):
            U_curr, kwargs = carry
            L = chol_softplus(U_curr)
            P_curr = L @ L.T
            m_curr = mean_update(P_curr, **kwargs)
            kwargs_curr = update_kwargs(m_curr)
            H_bar = kwargs_curr["H_bar"]
            S_t = H_bar @ P_curr @ H_bar.T + Rt
            W_curr = H_bar.T @ jnp.linalg.solve(S_t, H_bar)
            U_new = solve_inner_lbfgs(U_curr, kwargs_curr, W_curr=W_curr, max_inner_iters=self.n_inner)
            return U_new, kwargs_curr

        kwargs = update_kwargs(bel_pred.mean)
        bel_ekf = super()._update(bel_pred, y, x, kwargs["yhat"], kwargs["Ht"], Rt)
        U_init = (jnp.zeros_like(bel_pred.cov), update_kwargs(bel_ekf.mean))
        U_final = jax.lax.fori_loop(0, self.n_outer, body_fun, U_init)
        L = chol_softplus(U_final[0])
        P0 = L @ L.T
        cov_new = L_pred @ P0 @ L_pred.T
        mean_new = mean_update(P0, **U_final[1])
        bel_update = bel.replace(mean=mean_new, cov=cov_new)
        output = callback_fn(bel_update, bel_pred, y, x)
        return bel_update, output