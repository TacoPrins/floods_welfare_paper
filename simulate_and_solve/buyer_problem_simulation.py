"""
buyer_problem.py

Mortgage pricing: for ltv>0 the buyer receives q*L with L = ltv*lom_price (face value), where q solves the
same fixed point as in the household problem (misc.price_loan). The proportional origination cost is inside q.
"""

import numpy as np
import misc_functions as misc
from numba import njit


@njit(fastmath=True)
def solve(par, grids, curr_h_index, vX, j_index, dP, mVt_stayer, b_cols, q_cols, q_starts, dP_lom,max_ltv,max_ltv_index,selected_values_buy):
    # b_cols, q_cols, q_starts : stayer savings policy and mortgage price blended over the LTV grid, from
    #                            misc.ltv_tables, indexed [h, l]; l = vL_sim.size is the between-grid limit max_ltv[h]

    mVt = np.ones((grids.vM_sim.size))*-1e12
    mH_pol_index = np.zeros((grids.vM_sim.size),dtype=np.int64)
    mL_pol_index = np.zeros((grids.vM_sim.size),dtype=np.int64)
    mH_pol=np.zeros((grids.vM_sim.size),dtype=np.float64)
    mLTV_max= np.zeros(grids.vM_sim.size, dtype=np.bool_)


    for h_index in range(grids.vH.size):
        if h_index==curr_h_index:
            continue
        h=grids.vH[h_index]
        lom_price=dP_lom*h
        buy_price=(dP*(1+par.dKappa_buy))*h
        if j_index == par.iNj - 1:
            max_index=0
            max_it = 0
        else:
            max_index=max_ltv_index[h_index]
            #We add an iteration for the between-grid LTV limit
            max_it = max_index+1
        LTV_max=False
        for iteration in range(0,max_it+1):
            if iteration<=max_index:
                l_index_sim=iteration
                ltv = grids.vL_sim[l_index_sim]
            elif iteration>max_index and max_ltv[h_index]>grids.vL_sim[max_index]:
                l_index_sim=max_index
                ltv=max_ltv[h_index]
                LTV_max=True
            else:
                continue
            l_index_l=misc.binary_search(0, grids.vL.size, grids.vL,ltv)
            if ltv>0:
                L = ltv*lom_price
                l_tab = grids.vL_sim.size if LTV_max else l_index_sim
                b_col = b_cols[h_index, l_tab]
                q_col = q_cols[h_index, l_tab]
                q_start = q_starts[h_index, l_tab]
                v_mono = misc.col_nondecreasing(grids.vL, mVt_stayer[:,h_index, :], ltv, l_index_l)
            #For stability, we do not let available mortgage grow with dP_C.
            m_index_l=0
            for index in selected_values_buy:
                if ltv>0:
                    m_pre = vX[index] - buy_price - par.dZeta_fixed
                    # q <= q_start, so the value at the no-default cash bounds this candidate from above
                    # (valid when the stayer value rises with cash): skip the fixed point if it cannot win
                    m_bound = m_pre + q_start*L
                    if m_bound <= grids.vM[0]:
                        continue
                    if v_mono:
                        v_bound = misc._interp_2d(grids.vM, grids.vL, mVt_stayer[:,h_index, :], m_bound, ltv, misc.binary_search(0, grids.vM.size, grids.vM, m_bound), l_index_l)
                        if v_bound <= mVt[index]:
                            continue
                    _, m_buyer, _ = misc.price_loan_col(grids.vM, grids.vB, b_col, q_col, m_pre, L, q_start)
                else:
                    m_buyer = vX[index] - buy_price
                if m_buyer>grids.vM[0]:
                    m_index_l=misc.binary_search_sim(m_index_l, grids.vM.size, grids.vM,m_buyer)
                    Vt_candidate=misc._interp_2d(grids.vM,grids.vL,mVt_stayer[:,h_index, :],m_buyer, ltv, m_index_l,l_index_l)
                    if Vt_candidate > mVt[index]:
                        mVt[index] = Vt_candidate
                        mH_pol[index]= h
                        mH_pol_index[index] = h_index
                        mL_pol_index[index] = l_index_sim
                        mLTV_max[index]=LTV_max

    return mVt, mH_pol, mH_pol_index, mLTV_max, mL_pol_index