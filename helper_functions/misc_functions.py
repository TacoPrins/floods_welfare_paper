"""
misc_functions.py
"""

import numpy as np
import numba as nb
from numba import njit

###########################################################
### mAH =  DoubleGrid(vA, vH)
@njit
def DoubleGrid(vA, vH):
    """
    Returns an (iNa*iNh) x 2 matrix containing all possible combinations 
    of savings (first column) and house values (second column).
    """
    iNa = vA.shape[0]
    iNh = vH.shape[0]
    N   = iNa * iNh

    # Allocate the output grid
    mAH = np.empty((N, 2), dtype=vA.dtype)

    # Fill it: for each house-value index j and saving-index i,
    # place (vA[i], vH[j]) at row idx = j*iNa + i
    for j in range(iNh):
        for i in range(iNa):
            idx = j * iNa + i
            mAH[idx, 0] = vA[i]
            mAH[idx, 1] = vH[j]

    return mAH

###########################################################
### Interpolation functions
@njit
def lininterp_zero_crossing(x_min, y_min, x_max, y_max):
    # Calculate the x value where the line between (x1, y1) and (x2, y2) crosses y=0
    x_zero = x_min + (- y_min) * (x_max - x_min) / (y_max - y_min)
    return x_zero


@njit(fastmath=True)
def _interp_3d(grid1,grid2,grid3,value,xi1,xi2,xi3,j1,j2,j3):
    """ 3d interpolation for one point with known location
        
    Args:

        grid1 (numpy.ndarray): 1d grid
        grid2 (numpy.ndarray): 1d grid
        grid3 (numpy.ndarray): 1d grid
        value (numpy.ndarray): value array (3d)
        xi1 (double): input point
        xi2 (double): input point
        xi3 (double): input point
        j1 (int): location in grid 
        j2 (int): location in grid
        j3 (int): location in grid

    Returns:

        yi (double): output

    """

    # a. left/right
    nom_1_left = grid1[j1+1]-xi1
    nom_1_right = xi1-grid1[j1]

    nom_2_left = grid2[j2+1]-xi2
    nom_2_right = xi2-grid2[j2]

    nom_3_left = grid3[j3+1]-xi3
    nom_3_right = xi3-grid3[j3]

    # b. interpolation
    denom = (grid1[j1+1]-grid1[j1])*(grid2[j2+1]-grid2[j2])*(grid3[j3+1]-grid3[j3])
    nom = 0
    for k1 in range(2):
        nom_1 = nom_1_left if k1 == 0 else nom_1_right
        for k2 in range(2):
            nom_2 = nom_2_left if k2 == 0 else nom_2_right       
            for k3 in range(2):
                nom_3 = nom_3_left if k3 == 0 else nom_3_right               
                nom += nom_1*nom_2*nom_3*value[j1+k1,j2+k2,j3+k3]

    return nom/denom

@njit(fastmath=True)
def interp_3d(grid1,grid2,grid3,value,xi1,xi2,xi3):
    """ 3d interpolation for one point
        
    Args:

        grid1 (numpy.ndarray): 1d grid
        grid2 (numpy.ndarray): 1d grid
        grid3 (numpy.ndarray): 1d grid
        value (numpy.ndarray): value array (3d)
        xi1 (double): input point
        xi2 (double): input point
        xi3 (double): input point

    Returns:

        yi (double): output

    """

    # a. search in each dimension
    j1 = binary_search(0,grid1.size,grid1,xi1)
    j2 = binary_search(0,grid2.size,grid2,xi2)
    j3 = binary_search(0,grid3.size,grid3,xi3)

    return _interp_3d(grid1,grid2,grid3,value,xi1,xi2,xi3,j1,j2,j3)

@njit(fastmath=True)
def _interp_4d(grid1,grid2,grid3,grid4,value,xi1,xi2,xi3,xi4,j1,j2,j3,j4):
    """ 4d interpolation for one point with known location
        
    Args:

        grid1 (numpy.ndarray): 1d grid
        grid2 (numpy.ndarray): 1d grid
        grid3 (numpy.ndarray): 1d grid
        grid4 (numpy.ndarray): 1d grid
        value (numpy.ndarray): value array (4d)
        xi1 (double): input point
        xi2 (double): input point
        xi3 (double): input point
        xi4 (double): input point
        j1 (int): location in grid 
        j2 (int): location in grid
        j3 (int): location in grid
        j4 (int): location in grid

    Returns:

        yi (double): output

    """

    # a. left/right
    nom_1_left = grid1[j1+1]-xi1
    nom_1_right = xi1-grid1[j1]

    nom_2_left = grid2[j2+1]-xi2
    nom_2_right = xi2-grid2[j2]

    nom_3_left = grid3[j3+1]-xi3
    nom_3_right = xi3-grid3[j3]

    nom_4_left = grid4[j4+1]-xi4
    nom_4_right = xi4-grid4[j4]

    # b. interpolation
    denom = (grid1[j1+1]-grid1[j1])*(grid2[j2+1]-grid2[j2])*(grid3[j3+1]-grid3[j3])*(grid4[j4+1]-grid4[j4])
    nom = 0
    for k1 in range(2):
        nom_1 = nom_1_left if k1 == 0 else nom_1_right
        for k2 in range(2):
            nom_2 = nom_2_left if k2 == 0 else nom_2_right       
            for k3 in range(2):
                nom_3 = nom_3_left if k3 == 0 else nom_3_right  
                for k4 in range(2):
                    nom_4 = nom_4_left if k4 == 0 else nom_4_right  
                    nom += nom_1*nom_2*nom_3*nom_4*value[j1+k1,j2+k2,j3+k3,j4+k4]

    return nom/denom

@njit(fastmath=True)
def interp_4d(grid1,grid2,grid3,grid4,value,xi1,xi2,xi3,xi4):
    """ 4d interpolation for one point
        
    Args:

        grid1 (numpy.ndarray): 1d grid
        grid2 (numpy.ndarray): 1d grid
        grid3 (numpy.ndarray): 1d grid
        grid4 (numpy.ndarray): 1d grid
        value (numpy.ndarray): value array (4d)
        xi1 (double): input point
        xi2 (double): input point
        xi3 (double): input point
        xi4 (double): input point

    Returns:

        yi (double): output

    """

    # a. search in each dimension
    j1 = binary_search(0,grid1.size,grid1,xi1)
    j2 = binary_search(0,grid2.size,grid2,xi2)
    j3 = binary_search(0,grid3.size,grid3,xi3)
    j4 = binary_search(0,grid4.size,grid4,xi4)

    return _interp_4d(grid1,grid2,grid3,grid4,value,xi1,xi2,xi3,xi4,j1,j2,j3,j4)

#################
# binary search #
#################

@njit(fastmath=True)
def binary_search(imin,Nx,x,xi):
        
    # a. checks
    if xi <= x[0]:
        return 0
    elif xi >= x[Nx-2]:
        return Nx-2
    
    # b. binary search
    half = Nx//2
    while half:
        imid = imin + half
        if x[imid] <= xi:
            imin = imid
        Nx -= half
        half = Nx//2
        
    return imin

###########################################################
### vMaxRow, vArgMaxRow= maxRow(mMatrix)
@njit
def maxRow(mMatrix):
    """Returns the MaxRow of a matrix.

    Args:
        mMatrix (matrix, float): input matrix

    Returns:
        vMaxRow (vector, float): Maxrow vector
        vArgMaxRow (vector, integer): argument of max row in each col
    """
    iCols = mMatrix.shape[1]
    vMaxRow = np.zeros(iCols, dtype = nb.float64)
    vArgMaxRow = np.zeros(iCols, dtype = nb.int64)
    for i in range(iCols):
        vMaxRow[i] = np.max(mMatrix[:,i])
        vArgMaxRow[i] = np.argmax(mMatrix[:,i])
    return vMaxRow, vArgMaxRow

###########################################################
### par = construct_jitclass(dictionary):
def construct_jitclass(dictionary):
    """
    Convert dictionary into numba object.

    Args:
        dictionary (dictionary): Dictionary with all model parameters or other objects.

    Returns:
        par (numba object): Numba object with all model parameters or other objects.
    """

    parlist = [(key, nb.typeof(val)) for key, val in dictionary.items()]

    @nb.experimental.jitclass(parlist)
    class ParClass:
        def __init__(self):
            pass

    par = ParClass()
    for key, val in dictionary.items():
        setattr(par, key, val)

    return par

###########################################################
### 2D inteprolation

@njit(fastmath=True)
def _interp_2d(grid1,grid2,value,xi1,xi2,j1,j2):
    """ 2d interpolation for one point with known location
        
    Args:

        grid1 (numpy.ndarray): 1d grid
        grid2 (numpy.ndarray): 1d grid
        value (numpy.ndarray): value array (2d)
        xi1 (double): input point
        xi2 (double): input point
        j1 (int): location in grid 
        j2 (int): location in grid

    Returns:

        yi (double): output

    """

    # a. left/right
    nom_1_left = grid1[j1+1]-xi1
    nom_1_right = xi1-grid1[j1]

    nom_2_left = grid2[j2+1]-xi2
    nom_2_right = xi2-grid2[j2]

    # b. interpolation
    denom = (grid1[j1+1]-grid1[j1])*(grid2[j2+1]-grid2[j2])
    nom = 0
    for k1 in range(2):
        nom_1 = nom_1_left if k1 == 0 else nom_1_right
        for k2 in range(2):
            nom_2 = nom_2_left if k2 == 0 else nom_2_right                    
            nom += nom_1*nom_2*value[j1+k1,j2+k2]

    return nom/denom

@njit(fastmath=True)
def interp_2d(grid1,grid2,value,xi1,xi2):
    """ 2d interpolation for one point
        
    Args:

        grid1 (numpy.ndarray): 1d grid
        grid2 (numpy.ndarray): 1d grid
        value (numpy.ndarray): value array (2d)
        xi1 (double): input point
        xi2 (double): input point

    Returns:

        yi (double): output

    """

    # a. search in each dimension
    j1 = binary_search(0,grid1.size,grid1,xi1)
    j2 = binary_search(0,grid2.size,grid2,xi2)

    return _interp_2d(grid1,grid2,value,xi1,xi2,j1,j2)

#################
# binary search #
#################

@njit(fastmath=True)
def binary_search_sim(imin,Nx,x,xi):

    # a. checks
    if xi <= x[imin]:
        imin=binary_search(0,Nx,x,xi)
    elif xi >= x[Nx-2]:
        return Nx-2    
    elif xi <= x[imin+1]:
        return imin
    elif xi <= x[imin+2]:
        return imin+1
    else:
        imin=binary_search(0,Nx,x,xi) 
        
    return imin

@njit
def ols_numba(X, Y):
    """Compute OLS estimates using (X'X)^-1 X'Y, Numba optimized."""
    XtX = X.T @ X 
    XtY = X.T @ Y  
    beta = np.linalg.solve(XtX, XtY)  
    return beta


@njit 
def net_income(par, grids, j, e_index, e_trans_index, mortgage_size, mortgage_rate):
    if j<par.j_ret:
        pretax_income=np.exp(grids.vChi[j] + grids.vE[e_index]+ grids.vE_trans[e_trans_index])
    else:
        pretax_income=0.7*np.exp(grids.vChi[par.j_ret-1] + grids.vE[e_index])
    posttax_income=pretax_income-par.tau_0*(max(pretax_income-mortgage_rate*mortgage_size*grids.median_inc,0))**(1-par.tau_1)
    mortgage_rebate=par.tau_0*(pretax_income)**(1-par.tau_1)-(pretax_income-posttax_income)
    ##Normalise with pre-tax median income
    net_income=(posttax_income/grids.median_inc)*(1-par.wf_wedge[0])    
    mortgage_rebate=(mortgage_rebate/grids.median_inc)*(1-par.wf_wedge[0])
        
    return net_income, mortgage_rebate

###########################################################
### Mortgage pricing (KMV-style, household-specific q)
###########################################################
#
# q_{t,j}(b', h, l', e) = price per unit of face value that a borrower receives at origination.
# Face value L = l' * h * P_t. Cash received = q * L.
# q = (1 - dZeta) * [s * V^sceptic + (1 - s) * V^realist], where V is the lender's value of
# one unit of outstanding debt (=1 without default risk when contract rate = lender discount rate).
# V is computed in continuation_value (solve_owners_C / solve_owners_NC); pooling happens in
# household_problem. Households take q as given w.r.t. b' in the origination period, so each
# origination is a small fixed point: cash -> b' -> q -> cash.

@njit
def lender_discount_rate(par):
    """Lender's cost of funds, common to both locations.
    Baseline: equal to the inland contract rate, so V=1 without default risk.
    If the coastal contract rate differs (e.g. config.mortgage_premium), the coastal V
    without default risk is the PV of the payment stream at this rate, and can exceed 1."""
    return par.r_m_nc


###########################################################
### Fixed point on precomputed LTV columns (for the simulation's loops)
###########################################################
# In the simulation the LTV of a candidate loan is fixed while price_loan is called for many cash
# levels. Bilinear interpolation at a fixed LTV equals linear interpolation in the first dimension of
# the LTV-blended column, so blending once per (h, ltv) and interpolating in 1D gives the same numbers
# (up to rounding) at a fraction of the cost: no search over vL, and 2 instead of 4 grid points.

@njit
def ltv_column(vL, value_hl, ltv):
    """Blend value_hl[:, l] over the LTV grid at ltv (linear, extrapolating like interp_2d)."""
    l_idx = binary_search(0, vL.size, vL, ltv)
    w = (ltv - vL[l_idx]) / (vL[l_idx+1] - vL[l_idx])
    return (1-w)*value_hl[:, l_idx] + w*value_hl[:, l_idx+1]

@njit
def _interp_col(grid, col, x):
    i = binary_search(0, grid.size, grid, x)
    return col[i] + (x - grid[i])*(col[i+1] - col[i])/(grid[i+1] - grid[i])


@njit
def _q_map_col(vM, vB, b_col, q_col, m_pre, L, q):
    """step 2-5: compute m based on q (guess), find b' for that m. find q for that b'"""
    m = m_pre + q*L
    b = _interp_col(vM, b_col, m)
    b_clip = min(max(b, vB[0]), vB[-1])
    q_new = _interp_col(vB, q_col, b_clip)
    if q_new < 0.0:
        q_new = 0.0
    return q_new, b


@njit
def _bisect_fixed_point_col(vM, vB, b_col, q_col, m_pre, L, lo, hi, tol):
    for it in range(60):
        mid = 0.5*(lo + hi)
        f_mid, _ = _q_map_col(vM, vB, b_col, q_col, m_pre, L, mid)
        if mid - f_mid > 0.0:
            hi = mid
        else:
            lo = mid
        if (hi - lo)*L < tol:
            break
    return 0.5*(lo + hi)


@njit
def price_loan_col(vM, vB, b_col, q_col, m_pre, L, q_start, tol=1e-9, max_it=50):
    """the algorithm itself for b_col = ltv_column(vL, b_stay_hl, ltv), q_col = ltv_column(vL, q_hl, ltv),
    and q_start = q_col.max() (= q_upper). Same algorithm and stopping rules as price_loan."""
    q = q_start
    q_prev = q_start
    for it in range(max_it):
        q_new, b = _q_map_col(vM, vB, b_col, q_col, m_pre, L, q)
        if abs(q_new - q)*L < tol: # converged
            return q_new, m_pre + q_new*L, b
        if q_new > q: # algoritgm spirals, do a bisection on the grid
            q_root = _bisect_fixed_point_col(vM, vB, b_col, q_col, m_pre, L, q, q_prev, tol)
            _, b = _q_map_col(vM, vB, b_col, q_col, m_pre, L, q_root)
            return q_root, m_pre + q_root*L, b
        q_prev = q
        q = q_new
    # non-convergence with algorithm, do bisection
    q_root = _bisect_fixed_point_col(vM, vB, b_col, q_col, m_pre, L, 0.0, q, tol)
    _, b = _q_map_col(vM, vB, b_col, q_col, m_pre, L, q_root)
    return q_root, m_pre + q_root*L, b

@njit
def col_nondecreasing(vL, value_ml, ltv, l_idx):
    # for speed-ups
    """True if value_ml (vM, vL) blended over the LTV grid at ltv (inside the grid cell l_idx) never falls with
    cash. Only then does the value at the no-default cash bound a mortgage candidate from above."""
    w = (ltv - vL[l_idx]) / (vL[l_idx+1] - vL[l_idx])
    if w < 0.0 or w > 1.0:
        return False
    for m_index in range(value_ml.shape[0]-1):
        v0 = (1-w)*value_ml[m_index, l_idx] + w*value_ml[m_index, l_idx+1]
        v1 = (1-w)*value_ml[m_index+1, l_idx] + w*value_ml[m_index+1, l_idx+1]
        if v1 < v0:
            return False
    return True

@njit
def ltv_tables(grids, b_input, q_input, max_ltv):
    # used in simulation
    """LTV columns for every house size h and every LTV the simulation can choose: the points of
    grids.vL_sim (index l), plus the between-grid limit max_ltv[h] (index vL_sim.size).
    b_input: (vM, vH, vL) stayer savings policy, q_input: (vB, vH, vL) mortgage price, at (t, j, k, g, e).
    Returns b_cols (H, nL+1, vM), q_cols (H, nL+1, vB) and q_start = max of each q column."""
    nL = grids.vL_sim.size
    b_cols = np.zeros((grids.vH.size, nL+1, grids.vM.size))
    q_cols = np.zeros((grids.vH.size, nL+1, grids.vB.size))
    q_start = np.zeros((grids.vH.size, nL+1))
    for h_index in range(grids.vH.size):
        for l in range(nL+1):
            ltv = grids.vL_sim[l] if l < nL else max_ltv[h_index]
            b_cols[h_index, l] = ltv_column(grids.vL, b_input[:, h_index, :], ltv)
            q_cols[h_index, l] = ltv_column(grids.vL, q_input[:, h_index, :], ltv)
            q_start[h_index, l] = q_cols[h_index, l].max()
    return b_cols, q_cols, q_start

# Diagnostic only (not called by price_loan). Use offline on a few slices, e.g. to count
# how often q falls with savings; a violation is not an error.
@njit
def check_price_loan_monotonicity(vM, vB, vL, b_stay_hl, q_hl,
                                  ltv, check_tol=1e-8):
    """Sufficient test for monotonicity of F(q)=q(b_stay(m_pre+q*L)).

    At the chosen LTV, check that savings is nondecreasing in cash m,
    and the offered mortgage price is nondecreasing in savings b.
    These checks are exact for the piecewise-linear interpolation used
    by _q_map, including between-grid LTV choices.

    Returns: 0 if both checks pass, 1 if savings declines in m,
    2 if q declines in b. A failure is not proof that F is
    nonmonotone on the cash range relevant to a particular loan.
    """
    li = binary_search(0, vL.size, vL, ltv)
    weight = (ltv - vL[li]) / (vL[li+1] - vL[li])

    b_previous = (1-weight)*b_stay_hl[0,li] + weight*b_stay_hl[0,li+1]
    if not np.isfinite(b_previous):
        return 1
    for i in range(1, vM.size):
        b_current = (1-weight)*b_stay_hl[i,li] + weight*b_stay_hl[i,li+1]
        if not np.isfinite(b_current) or b_current < b_previous - check_tol:
            return 1
        b_previous = b_current

    q_previous = (1-weight)*q_hl[0,li] + weight*q_hl[0,li+1]
    if not np.isfinite(q_previous):
        return 2
    for i in range(1, vB.size):
        q_current = (1-weight)*q_hl[i,li] + weight*q_hl[i,li+1]
        if not np.isfinite(q_current) or q_current < q_previous - check_tol:
            return 2
        q_previous = q_current

    return 0