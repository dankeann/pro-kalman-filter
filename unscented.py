import jax
import jax.numpy as jnp
from profilter import ExtendedKalmanFilter


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