"""
mortgage_choice_simulation.py

Refinancing (l_choice above the minimum-payment LTV) is priced KMV-style: the old loan is repaid and the new
loan pays q*L in cash, with q from the same fixed point as in the household problem (misc.price_loan).
"""

import numpy as np
import misc_functions as misc
import interp as interp
from numba import njit

@njit(fastmath=True)
def solve(par,grids,vt_stay_input,b_cols,q_cols,q_starts,j,h,e, dZ, dP,mortgage_start,max_ltv_cache_choice_index_left, min_payment, ltv_cache_minpay_index_left, selected_values_stay, mortgage_rate):
    # b_cols, q_cols, q_starts : stayer savings policy and mortgage price for this h, blended over the LTV grid
    #                            at each vL_sim point (misc.ltv_tables[h_index])

    #Initialise value from paying off more than the minimum and refinancing
    vt_stay=np.ones((grids.vM_sim.size))*-1e12
    ltv_out=np.empty((grids.vM_sim.size))
    m_out=np.empty((grids.vM_sim.size))

    #Some helper variables to prevent unnecessary computations
    mortgage_withint=(1+mortgage_rate)*mortgage_start
    house_value=h*dP
    depreciation=((1-dZ)+par.dDelta+par.dTau_h)*house_value

    #cash in hand after housing costs
    m_0=grids.vM[0]

    if ltv_cache_minpay_index_left>max_ltv_cache_choice_index_left:
        max_ltv_cache_choice_index_left=ltv_cache_minpay_index_left

    if j<par.iNj-1:
        # Value of paying the minimum (filled in below as before); used here only to skip refinancing
        # candidates that cannot beat it
        minpay_value=np.ones((grids.vM_sim.size))*-1e12
        ltv_minpay =(mortgage_withint-min_payment)/(house_value)
        l_index_minpay=misc.binary_search(0, grids.vL.size, grids.vL,ltv_minpay)
        payment_minpay=mortgage_withint-ltv_minpay*house_value
        for m_index_sim in selected_values_stay:
            stayer_cih=grids.vM_sim[m_index_sim]+e-depreciation-payment_minpay
            if stayer_cih>m_0:
                minpay_value[m_index_sim]=misc._interp_2d(grids.vM, grids.vL, vt_stay_input, stayer_cih, ltv_minpay, misc.binary_search(0, grids.vM.size, grids.vM, stayer_cih), l_index_minpay)

        l_index_l=0
        for l_choice_index in range(0,max_ltv_cache_choice_index_left+1):
            ltv_cache = grids.vL_sim[l_choice_index]
            l_index_l=misc.binary_search_sim(l_index_l, grids.vL.size, grids.vL,ltv_cache)
            refinance = l_choice_index>ltv_cache_minpay_index_left
            payment = 0.
            L_new = 0.
            q_start = 0.
            b_col = b_cols[l_choice_index]
            q_col = q_cols[l_choice_index]
            if refinance:
                L_new = ltv_cache*house_value
                q_start = q_starts[l_choice_index]
                v_mono = misc.col_nondecreasing(grids.vL, vt_stay_input, ltv_cache, l_index_l)
            else:
                payment=mortgage_withint-ltv_cache*house_value
            m_index_l=0
            for m_index_sim in selected_values_stay:
                if refinance:
                    # old loan repaid, new loan pays q*L_new; q from the price-taking fixed point
                    m_pre = grids.vM_sim[m_index_sim]+e-depreciation-mortgage_withint-par.dZeta_fixed
                    # q <= q_start, so the value at the no-default cash bounds this candidate from above
                    # (valid when the stayer value rises with cash): skip the fixed point if it cannot win
                    m_bound = m_pre + q_start*L_new
                    if m_bound <= m_0:
                        continue
                    if v_mono:
                        v_bound = misc._interp_2d(grids.vM, grids.vL, vt_stay_input, m_bound, ltv_cache, misc.binary_search(0, grids.vM.size, grids.vM, m_bound), l_index_l)
                        if v_bound <= vt_stay[m_index_sim] or v_bound < minpay_value[m_index_sim]:
                            continue
                    _, stayer_cih, _ = misc.price_loan_col(grids.vM, grids.vB, b_col, q_col, m_pre, L_new, q_start)
                else:
                    stayer_cih=grids.vM_sim[m_index_sim]+e-depreciation-payment
                if stayer_cih>m_0:
                    m_index_l=misc.binary_search_sim(m_index_l, grids.vM.size, grids.vM,stayer_cih)
                    candidate_value=misc._interp_2d(grids.vM, grids.vL, vt_stay_input, stayer_cih, ltv_cache,m_index_l, l_index_l)
                    if candidate_value>vt_stay[m_index_sim]:
                        m_out[m_index_sim]=stayer_cih
                        ltv_out[m_index_sim] = ltv_cache
                        vt_stay[m_index_sim]=candidate_value
        ltv_cache =(mortgage_withint-min_payment)/(house_value)
        l_index_l=misc.binary_search(0, grids.vL.size, grids.vL,ltv_cache)
        payment=mortgage_withint-ltv_cache*house_value
        m_index_l=0
        for m_index_sim in selected_values_stay:
            stayer_cih=grids.vM_sim[m_index_sim]+e-depreciation-payment
            if stayer_cih>m_0:
                m_index_l=misc.binary_search_sim(m_index_l, grids.vM.size, grids.vM,stayer_cih)
                candidate_value=misc._interp_2d(grids.vM, grids.vL, vt_stay_input, stayer_cih, ltv_cache,m_index_l, l_index_l)
                if candidate_value>vt_stay[m_index_sim]:
                    m_out[m_index_sim]=stayer_cih
                    ltv_out[m_index_sim] = ltv_cache
                    vt_stay[m_index_sim]=candidate_value
    else:
        payment=mortgage_withint
        m_index_l=0
        for m_index_sim in selected_values_stay:
            stayer_cih=grids.vM_sim[m_index_sim]+e-depreciation-payment
            m_index_l=misc.binary_search_sim(m_index_l, grids.vM.size, grids.vM,stayer_cih)
            if stayer_cih>m_0:
                candidate_value=interp._interp_1d(grids.vM, vt_stay_input[:,0], stayer_cih,m_index_l)
                if candidate_value>vt_stay[m_index_sim]:
                    m_out[m_index_sim]=stayer_cih
                    ltv_out[m_index_sim] = 0
                    vt_stay[m_index_sim]=candidate_value


    return vt_stay, ltv_out, m_out