"""
Tauchen.py

Purpose:
    Discretise the persistent AR(1) component of log income and build the income distribution
    over the life cycle.
"""
###########################################################
### Imports
import numpy as np
from scipy.stats import norm

###########################################################
### Functions
###########################################################

def rouwenhorst(dRho, dSigmaeps, iNumStates, time_increment):
    """Returns the markov chain and grid of an AR(1) process using the Rouwenhorst method

    The process is z' = dRho * z + eps, eps ~ N(0, dSigmaeps^2), with annual parameters.  Unlike
    Tauchen, Rouwenhorst matches the unconditional variance and the autocorrelation exactly, also
    for persistence close to one and few states (Kopecky & Suen 2010).  The annual transition
    matrix is raised to the power time_increment for multi-year model periods.

    Args:
        dRho           (float): Persistence of the AR(1), annual
        dSigmaeps      (float): Standard deviation of the error term, annual
        iNumStates     (int):   Number of grid points
        time_increment (int):   Years per model period

    Returns:
        mMarkov (matrix, float): Markov chain over one model period (rows: current state)
        vZ      (vector, float): Grid, evenly spaced on +/- sd(z) * sqrt(iNumStates - 1)
    """
    p = (1 + dRho) / 2
    mMarkov = np.array([[p, 1 - p], [1 - p, p]])
    for n in range(3, iNumStates + 1):
        mNew = np.zeros((n, n))
        mNew[:-1, :-1] += p * mMarkov
        mNew[:-1, 1:]  += (1 - p) * mMarkov
        mNew[1:, :-1]  += (1 - p) * mMarkov
        mNew[1:, 1:]   += p * mMarkov
        mNew[1:-1, :]  /= 2                      # interior rows are counted twice
        mMarkov = mNew

    dSD_Z = dSigmaeps / np.sqrt(1 - dRho**2)
    dZ_max = dSD_Z * np.sqrt(iNumStates - 1)
    vZ = np.linspace(-dZ_max, dZ_max, iNumStates)

    return np.linalg.matrix_power(mMarkov, time_increment), vZ


def tauchen(dRho, dSigmaeps, iNumStates, iM, time_increment):
    """Returns the markov chain and grid of an AR(1) process using the Tauchen method
    (superseded by rouwenhorst, which is more accurate for persistence close to one)

    Args:
        dRho        (float): Persistence of the AR(1)
        dSigmaeps   (float): Standard deviation of the error term in the AR(1)
        iNumStates  (int):   Number of grid points for the discrete state space
        iM          (int):   Width of the grid in unconditional standard deviations

    Returns:
        mMarkov (matrix, float): Markov chain
        vZ      (vector, float): Grid
    """
    dSD_Z = np.sqrt(dSigmaeps**2 / (1 - dRho**2))
    vZ = np.linspace(-iM * dSD_Z, iM * dSD_Z, iNumStates)
    dDist = vZ[1] - vZ[0]

    # Probability of landing in each interval around the grid points; outer intervals are open
    vUpper = np.append(vZ[:-1] + 0.5 * dDist, np.inf)
    vLower = np.append(-np.inf, vZ[1:] - 0.5 * dDist)
    mMean = dRho * vZ[:, np.newaxis]
    mMarkov = norm.cdf((vUpper - mMean) / dSigmaeps) - norm.cdf((vLower - mMean) / dSigmaeps)

    return np.linalg.matrix_power(mMarkov, time_increment), vZ


def initial_dist(par, vE):
    """Distribution of entrants over the income states: N(0, sd_income_initial^2) binned at the
    midpoints between grid points (outer bins open-ended)."""
    vCdf = norm.cdf(0.5 * (vE[:-1] + vE[1:]) / par.sd_income_initial)
    return np.diff(np.concatenate(([0.0], vCdf, [1.0])))


def weight_matrix(par, vE, vPi_E, mMarkov):
    """Distribution over income states by age: rolled forward with the Markov chain while working,
    frozen after retirement."""
    mPi_E = np.zeros((par.iNj, vE.size))
    mPi_E[0, :] = vPi_E
    for j in range(1, par.iNj):
        mPi_E[j, :] = mPi_E[j - 1, :] @ mMarkov if j < par.j_ret else mPi_E[j - 1, :]
    return mPi_E


def lifecycle(par, J_ret):
    """Age component of log income: fourth-order polynomial in age, for working-age periods."""
    vAge = 21 + par.time_increment * np.arange(J_ret)
    return (par.beta0_age + par.beta1_age * vAge + par.beta2_age * vAge**2
            + par.beta3_age * vAge**3 + par.beta4_age * vAge**4)


def median_inc(chi, z, z_dist):
    """Median of pretax income across working ages (equal weight per age) and income states."""
    vIncome = np.exp(chi[:, np.newaxis] + z[np.newaxis, :]).ravel()
    vDens = (z_dist[:chi.size, :] / chi.size).ravel()

    order = np.argsort(vIncome)
    vCumDens = np.cumsum(vDens[order]) / np.sum(vDens)
    return np.interp(0.5, vCumDens, vIncome[order])


def invar_dist(mMarkov):
    """Stationary distribution of a Markov chain."""
    evals, evecs = np.linalg.eig(mMarkov.T)
    evec1 = (evecs[:, np.isclose(evals, 1)])[:, 0]
    return evec1 / evec1.sum()


def combine_vectors(vE, vE_trans, vE_stationary, vE_trans_stationary):
    """Combine persistent and transitory states (all sums) and their probabilities (products)."""
    sum_vec = (vE[:, np.newaxis] + vE_trans[np.newaxis, :]).ravel()
    prob_vec = (vE_stationary[:, np.newaxis] * vE_trans_stationary[np.newaxis, :]).ravel()
    return sum_vec, prob_vec