"""
buyer_problem.py

Mortgage pricing (KMV-style): a buyer with LTV l>0 receives q*L in cash against face value L=ltv*dP*h.
q depends on the savings b' the buyer ends up with, so for every (x, h, l>0) we solve the small fixed
point cash -> b' -> q -> cash (misc.price_loan). Households take q as given w.r.t. b' (price-taking).
The proportional origination cost dZeta is inside q; only the fixed fee dZeta_fixed is charged here.
"""

import numpy as np
from numba import njit
import utility_epsilons as ut
import interp as interpfun
import misc_functions as misc

@njit
def solve(par, grids, j_index, k_index, g_index, e_index, dP, max_mortgage_pti, mVt_stayer, mC_pol_stayer, mB_pol_stayer, mQm, mVt_stayer_wf, config):
    # mQm           : (vB, vH, vL) pooled mortgage price q_{t,j}(b', h, l', e) for this g,k guy

    j=j_index
    g=grids.vG[g_index]

    mVt = np.ones((grids.vX.size))*-1e12
    mVt_wf = np.ones((grids.vX.size))*-1e12
    mQt = np.zeros((grids.vX.size))

    # Starting value of the fixed point (price without default risk), per (h, l). Needs max over all b' choices
    mQ_start = np.zeros((grids.vH.size, grids.vL.size))
    for h_index in range(grids.vH.size):
        for l_index in range(grids.vL.size):
            mQ_start[h_index, l_index] = mQm[:, h_index, l_index].max()

    # speeding up trick: Columns of the stayer value that never fall with cash (only there the bound below is valid)
    v_mono = np.zeros((grids.vH.size, grids.vL.size), dtype=np.bool_)
    for h_index in range(grids.vH.size):
        for l_index in range(grids.vL.size):
            v_mono[h_index, l_index] = np.all(mVt_stayer[1:, h_index, l_index] >= mVt_stayer[:-1, h_index, l_index])

    for x_index in range(grids.vX.size):
        x=grids.vX[x_index]
        for h_index in range(grids.vH.size):
            h=grids.vH[h_index]
            for l_index in range(grids.vL.size):
                ltv = grids.vL[l_index]
                if (l_index>0 and j_index==par.iNj-1) or ltv>grids.max_ltv:
                    continue
                if ltv*dP*h < max_mortgage_pti:     # PTI constraint on face value
                    if l_index==0:
                        buyer_cost=(dP*(1+par.dKappa_buy))*h
                        m_buyer= x - buyer_cost
                    else:
                        L = ltv*dP*h
                        m_pre = x - (dP*(1+par.dKappa_buy))*h - par.dZeta_fixed
                        # q <= q_start, so the value at the no-default cash bounds this candidate from above
                        # (valid when the stayer value rises with cash): skip the fixed point if it cannot win
                        m_bound = m_pre + mQ_start[h_index, l_index]*L
                        if m_bound < grids.vM[0]:
                            continue
                        # check whether the stayer value at the bound is below the current candidate value; if so, skip the fixed point
                        if v_mono[h_index, l_index]:
                            if m_bound <= grids.vM[-1]:
                                v_bound = interpfun.interp_1d(grids.vM, mVt_stayer[:, h_index, l_index], m_bound)
                            else:
                                v_bound = -1/(-1/mVt_stayer[-1,h_index, l_index]+ut.u(j,mC_pol_stayer[-1,h_index, l_index]+m_bound-grids.vM[-1],h,g+par.dOmega, par)-ut.u(j,mC_pol_stayer[-1,h_index, l_index],h,g+par.dOmega, par))
                            if v_bound <= mVt[x_index]:
                                continue
                        # ltv is a grid point, so the LTV columns are plain slices: the algorithm in vectorized version
                        q_loan, m_buyer, _ = misc.price_loan_col(grids.vM, grids.vB, mB_pol_stayer[:, h_index, l_index], mQm[:, h_index, l_index], m_pre, L, mQ_start[h_index, l_index])
                    if m_buyer>=grids.vM[0] and m_buyer<=grids.vM[-1]:
                        Vt_candidate=interpfun.interp_1d(grids.vM,mVt_stayer[:,h_index, l_index],m_buyer)
                        if Vt_candidate > mVt[x_index]:
                            C_pol=interpfun.interp_1d(grids.vM,mC_pol_stayer[:,h_index, l_index],m_buyer)
                            assert C_pol > 0
                            mVt[x_index] = Vt_candidate
                            mQt[x_index] = -1/ut.u_c(j,C_pol,h,g+par.dOmega, par)
                            if config.welfare == True:
                                mVt_wf[x_index] = interpfun.interp_1d(grids.vM,mVt_stayer_wf[:,h_index, l_index],m_buyer)

                    if m_buyer>grids.vM[-1]:
                        Vt_candidate=-1/(-1/mVt_stayer[-1,h_index, l_index]+ut.u(j,mC_pol_stayer[-1,h_index, l_index]+m_buyer-grids.vM[-1],h,g+par.dOmega, par)-ut.u(j,mC_pol_stayer[-1,h_index, l_index],h,g+par.dOmega, par))
                        if Vt_candidate > mVt[x_index]:
                            C_pol=mC_pol_stayer[-1,h_index, l_index]+m_buyer-grids.vM[-1]
                            assert C_pol > 0
                            mVt[x_index] = Vt_candidate
                            mQt[x_index] = -1/ut.u_c(j,C_pol,h,g+par.dOmega, par)
                            if config.welfare == True:
                                mVt_wf[x_index] = -1/(-1/mVt_stayer_wf[-1,h_index, l_index]+ut.u(j,C_pol,h,g+par.dOmega, par)-ut.u(j,mC_pol_stayer[-1,h_index, l_index],h,g+par.dOmega, par))


    assert np.isnan(mVt).sum() == 0
    assert np.isnan(mQt).sum() == 0

    return mVt, mQt, mVt_wf