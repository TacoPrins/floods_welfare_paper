import numpy as np
import math
from numba import njit

@njit
def norm_cdf(x, mu, sigma):
    return 0.5 * (1.0 + math.erf((x-mu)/(sigma*math.sqrt(2.0))))

@njit
def initial_joint(par, grids, bequest):
    """Joint distribution of entrants over initial cash-on-hand (grids.vX_sim) and income states (grids.vE).

    - Zero wealth with a probability given by the logit in log income (in sds from the median).
    - Otherwise log wealth is normal, with the empirical variance (var_poswealth) and covariance with
      log income (from corr_poswealth_income).  Its mean is set so that average wealth across all
      entrants equals the bequest.  For a lognormal, E[W] = exp(mean of log W + var/2), so the mean
      of log wealth is log(mean wealth) - var/2.
    - Probabilities are binned at the midpoints (in logs) between grid points; the last bin is open-ended.
    """
    # Log wealth given log income z (both conditional on wealth > 0): mean + slope*z, residual variance
    cov_inc_poswealth = par.corr_poswealth_income*par.sd_income_initial*np.sqrt(par.var_poswealth)
    slope = cov_inc_poswealth/par.sd_income_initial**2
    cond_sd_poswealth = np.sqrt(par.var_poswealth - slope*cov_inc_poswealth)

    # Probability of zero wealth by income state, and on average
    vnowealth_frac = 1/(1 + np.exp(-(par.beta0_nowealth + par.beta1_nowealth*grids.vE/par.sd_income_initial)))
    dnowealth_frac = np.dot(grids.vPi_E, vnowealth_frac)

    # Mean of log wealth by income state, such that mean wealth among those with wealth > 0 is bequest/(1 - dnowealth_frac)
    vcond_mean_poswealth = np.log(bequest/(1 - dnowealth_frac)) - 0.5*par.var_poswealth + slope*grids.vE

    mPi_joint = np.zeros((grids.vX_sim.size, grids.vE.size))
    for e_index in range(grids.vE.size):
        dCdf_prev = 0.0
        for x_index_sim in range(1, grids.vX_sim.size):
            if x_index_sim < grids.vX_sim.size - 1:
                dUpper = 0.5*np.log(grids.vX_sim[x_index_sim]) + 0.5*np.log(grids.vX_sim[x_index_sim+1])
                dCdf = norm_cdf(dUpper, vcond_mean_poswealth[e_index], cond_sd_poswealth)
            else:
                dCdf = 1.0
            mPi_joint[x_index_sim, e_index] = (dCdf - dCdf_prev)*(1 - vnowealth_frac[e_index])*grids.vPi_E[e_index]
            dCdf_prev = dCdf

    # Zero-wealth households at the first grid point (vX_sim[0] = 0)
    mPi_joint[0, :] = grids.vPi_E*vnowealth_frac

    return mPi_joint