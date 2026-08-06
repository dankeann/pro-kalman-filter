import jax
import jax.numpy as jnp
import jaxopt
from profilter import ExtendedKalmanFilter
from profilter import chol_softplus



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


# Methods copy and pasted from profilter.py - UKFPrOFilter
class UKFPrOFilter(UnscentedKalmanFilter):
    def __init__(
        self, fn_latent, fn_obs, dynamics_covariance, observation_covariance, n_inner, n_outer,
    ):
        super().__init__(fn_latent, fn_obs, dynamics_covariance, observation_covariance)
        self.n_inner = n_inner
        self.n_outer = n_outer

    def step(self, bel, y, x, callback_fn):
        bel_pred = super()._predict(bel)
        L_pred = jnp.linalg.cholesky(bel_pred.cov)
        Ht = self.jac_obs(bel_pred.mean, x)
        Rt = self.observation_covariance
        yhat = self.vobs_fn(bel_pred.mean, x)
        At = Ht @ bel_pred.cov @ Ht.T
        H_bar = Ht @ L_pred

        U_init = jnp.zeros_like(bel_pred.cov)

        def mean_update(P0):
            S = H_bar @ P0 @ H_bar.T + Rt
            dm = bel_pred.cov @ Ht.T @ jnp.linalg.solve(S + Ht @ bel_pred.cov @ Ht.T, y-yhat)
            return bel_pred.mean + dm

        def mm_obj(U, W_curr):
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
                return mm_obj(U, W_curr=W_curr)

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
            U_curr = carry
            L = chol_softplus(U_curr)
            P_curr = L @ L.T
            S_t = H_bar @ P_curr @ H_bar.T + Rt
            W_curr = H_bar.T @ jnp.linalg.solve(S_t, H_bar)
            U_new = solve_inner_lbfgs(U_curr, W_curr=W_curr, max_inner_iters=self.n_inner)
            return U_new

        U_init = jnp.zeros_like(bel_pred.cov)
        U_final = jax.lax.fori_loop(0, self.n_outer, body_fun, U_init)
        L = chol_softplus(U_final)
        P0 = L @ L.T
        cov_new = L_pred @ P0 @ L_pred.T
        mean_new = mean_update(P0)
        bel_update = bel.replace(mean=mean_new, cov=cov_new)
        output = callback_fn(bel_update, bel_pred, y, x)
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