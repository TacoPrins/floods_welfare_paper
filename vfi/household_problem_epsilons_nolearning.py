"""
VFI.py

Purpose:
    Solve for policy and value functions for a given law of motions

Mortgage pricing (KMV-style, household-specific q):
    - solve_owners_C/NC return, besides the continuation values, the lender value V^k_{t,j}(b',h,l',e)
      of one unit of face value of a loan chosen at age j, date t, by a type-k borrower.
    - Sceptics may switch to realist between t and t+1 (simulation: grids.mTypes). Lenders know this:
      V^sceptic <- (1-frac_t) V^sceptic + frac_t V^realist (same state, both from t+1 policies).
    - Lenders do not observe the belief type. They pool with the exogenous share of sceptics,
      which is grids.mTypes[t,1] in every cohort and every g:
            q_{t,j}(g) = (1-dZeta) * [ s_t V^sceptic + (1-s_t) V^realist ]
    - The buyer at (t,j) uses q_{t,j}; the refinancer (inside the continuation value) uses q_{t+1,j+1}.
    Loop order is therefore: date t (backward) -> age j (backward) -> types (k,g).
    Within (t,j): continuation + stayer + renter for all types, then pooling, then buyers.
"""
###########################################################
### Imports
import numpy as np
import continuation_value_nolearning as continuation_value_epsilons
import stayer_problem as stayer_problem
import stayer_problem_renter as stayer_problem_renter
import buyer_problem_epsilons as buyer_problem_epsilons
from numba import njit, prange
import LoM_epsilons as lom
import misc_functions as misc

# Optional: solve the belief x amenity types of each (date, age) in parallel on all cores (Numba prange). The types are
# independent given the (t+1) objects, and each writes only its own [.., k, g, ..] slices, so results do not change.
# Off by default: in our tests it was slower than one core (the types share memory-heavy work) and compiles longer.
# Worth one timing try on your own machine; the number of cores used is set by the environment variable NUMBA_NUM_THREADS.
PARALLEL = False

###########################################################
### Functions
###########################################################


@njit(parallel=PARALLEL)
def solve(grids, par, vCoeff_C,vCoeff_NC, config):
    if config.sceptics==False:
        k_dim=1
    else:
        k_dim=grids.vK.size

    if config.run_experiment:
        t_index_start=int((par.experiment_year-par.starting_year)/par.time_increment)
    else:
        t_index_start=0

    shape_stay                          = (grids.vTime.size, par.iNj, k_dim, grids.vG.size, grids.vM.size, grids.vH.size, grids.vL.size, grids.vE.size)
    shape_buy                           = (grids.vTime.size, par.iNj, k_dim, grids.vG.size, grids.vX.size,  grids.vE.size)
    shape_rent                          = (grids.vTime.size, par.iNj, k_dim, grids.vG.size, grids.vX.size, grids.vE.size)
    # Mortgage price (pooled over belief and amenity types, so no k and no g dimension) and rolling lender values (2 dates)
    shape_qm                            = (grids.vTime.size, par.iNj, grids.vB.size, grids.vH.size, grids.vL.size, grids.vE.size)
    shape_lv                            = (2, par.iNj, k_dim, grids.vG.size, grids.vB.size, grids.vH.size, grids.vL.size, grids.vE.size)

    # Arrays that are only read at dates t and t+1 are kept for 2 dates only (rolling slots, like the lender values):
    # marginal values qt_stay, consumption c and the stayer welfare values vt_stay_wf. Index: slot = t_index % 2.
    shape_roll                          = (2, par.iNj, k_dim, grids.vG.size, grids.vM.size, grids.vH.size, grids.vL.size, grids.vE.size)

    #Welfare (_wf) arrays are only recorded when config.welfare is on; otherwise they are size-1 placeholders that are never filled
    if config.welfare:
        shape_stay_wf = shape_stay
        shape_buy_wf  = shape_buy
        shape_rent_wf = shape_rent
    else:
        shape_stay_wf = (1, 1, 1, 1, 1, 1, 1, 1)
        shape_buy_wf  = (1, 1, 1, 1, 1, 1)
        shape_rent_wf = (1, 1, 1, 1, 1, 1)

    vt_buy_c        = np.zeros(shape_buy, dtype = np.float64)
    vt_buy_c_wf      = np.zeros(shape_buy_wf, dtype = np.float64)
    qt_buy_nc       = np.zeros(shape_buy, dtype = np.float64)
    vt_stay_c       = np.zeros(shape_stay, dtype = np.float64)
    vt_stay_c_wf    = np.zeros(shape_roll, dtype = np.float64)
    qt_stay_c       = np.zeros(shape_roll, dtype = np.float64)
    vt_buy_nc       = np.zeros(shape_buy, dtype = np.float64)
    vt_buy_nc_wf    = np.zeros(shape_buy_wf, dtype = np.float64)
    qt_buy_c        = np.zeros(shape_buy, dtype = np.float64)
    vt_stay_nc      = np.zeros(shape_stay, dtype = np.float64)
    vt_stay_nc_wf   = np.zeros(shape_roll, dtype = np.float64)
    qt_stay_nc      = np.zeros(shape_roll, dtype = np.float64)
    vt_renter  = np.zeros(shape_rent, dtype = np.float64)
    vt_renter_wf= np.zeros(shape_rent_wf, dtype = np.float64)
    qt_renter  = np.zeros(shape_rent, dtype = np.float64)
    b_stay_c        = np.zeros(shape_stay, dtype = np.float64)
    b_stay_nc       = np.zeros(shape_stay, dtype = np.float64)
    b_renter        = np.zeros(shape_rent, dtype = np.float64)
    c_c             = np.zeros(shape_roll, dtype = np.float64)
    c_nc            = np.zeros(shape_roll, dtype = np.float64)
    qm_c            = np.ones(shape_qm, dtype = np.float64)*(1-par.dZeta)
    qm_nc           = np.ones(shape_qm, dtype = np.float64)*(1-par.dZeta)
    lv_c            = np.ones(shape_lv, dtype = np.float64)   #V=1: no-default value; also the (irrelevant) value at the last age
    lv_nc           = np.ones(shape_lv, dtype = np.float64)

    #Matrix of expected welfare on a savings (before interest) grid
    v_owner_c_wf=np.zeros(shape_stay_wf, dtype = np.float64)
    v_owner_nc_wf=np.zeros(shape_stay_wf, dtype = np.float64)
    v_nonowner_wf=np.zeros(shape_rent_wf, dtype = np.float64)

    #The household functions always expect welfare inputs. The stayer welfare values are always kept (rolling, cheap);
    #when welfare is off, the other welfare inputs are the ordinary value functions, and welfare outputs are discarded (not recorded).
    in_stay_c_wf, in_stay_nc_wf = vt_stay_c_wf, vt_stay_nc_wf
    if config.welfare:
        in_renter_wf, in_buy_c_wf, in_buy_nc_wf = vt_renter_wf, vt_buy_c_wf, vt_buy_nc_wf
    else:
        in_renter_wf, in_buy_c_wf, in_buy_nc_wf = vt_renter, vt_buy_c, vt_buy_nc

    mortgage_size_C = np.zeros((grids.vH.size,grids.vL.size), dtype = np.float64)
    mortgage_size_NC= np.zeros((grids.vH.size,grids.vL.size), dtype = np.float64)
    type_mat = misc.DoubleGrid(np.linspace(0, k_dim-1, k_dim), np.linspace(0, grids.vG.size-1, grids.vG.size))


    g_index_nc=int((grids.vG.size-1)/2)
    assert grids.vG[g_index_nc]==1

    if config.building_rest == True:
        arrival_rate_discount = (1-par.dDelta)**(grids.vTime-t_index_start) + (1-(1-par.dDelta)**(grids.vTime-t_index_start))*(1-par.dMitigation_rate)
    else:
        arrival_rate_discount = np.ones((grids.vTime.size))

    for t_index in range(grids.vTime.size-1, t_index_start-1, -1):
        dP_C = lom.LoM(par,grids,t_index,vCoeff_C)
        dP_NC = lom.LoM(par,grids,t_index,vCoeff_NC)
        t_index_prime=min(t_index+1,grids.vTime.size-1)
        if t_index==grids.vTime.size-1:
            dPi_S=grids.vPi_S_median[t_index]*arrival_rate_discount[t_index]
            dPi_L=grids.vPi_L[t_index]
            dP_C_prime = lom.LoM(par,grids,t_index,vCoeff_C)
            dP_NC_prime = lom.LoM(par,grids,t_index,vCoeff_NC)
        else:
            dPi_S=grids.vPi_S_median[t_index+1]*arrival_rate_discount[t_index+1]
            dPi_L=grids.vPi_L[t_index+1]
            dP_C_prime = lom.LoM(par,grids,t_index+1, vCoeff_C)
            dP_NC_prime = lom.LoM(par,grids,t_index+1,vCoeff_NC)
        for h_index in range(grids.vH.size):
            for l_index in range(grids.vL.size):
                mortgage_size_C[h_index,l_index]=grids.vL[l_index]*grids.vH[h_index]*dP_C
                mortgage_size_NC[h_index,l_index]=grids.vL[l_index]*grids.vH[h_index]*dP_NC

        # Rolling slots for the lender value: write slot (date t), read slot_next (date t+1).
        # In the terminal date the economy is stationary, so V_{T-1,j} uses V_{T-1,j+1} (same slot).
        slot = t_index % 2
        if t_index==grids.vTime.size-1:
            slot_next = slot
        else:
            slot_next = (t_index+1) % 2

        # Exogenous sceptic share (same in every cohort and every g) and switching rate between t and t+1
        share_sceptic = 0.
        frac_switch = 0.
        if k_dim==2:
            share_sceptic = grids.mTypes[t_index,1]
            if t_index<grids.vTime.size-1 and grids.mTypes[t_index,1]>1e-8:
                frac_switch = -(grids.mTypes[t_index+1,1]-grids.mTypes[t_index,1])/grids.mTypes[t_index,1]

        for j in range(par.iNj-1, -1, -1):
            #####################################################################
            # (1) Continuation values, lender values, stayers and renters, all types
            #####################################################################
            for type_index in prange(type_mat.shape[0]):
                _types_step1(type_index, type_mat, j, par, grids, dPi_S, dPi_L, dP_C_prime, mortgage_size_C, config, dP_NC_prime, mortgage_size_NC, shape_stay, t_index, vt_stay_c, in_stay_c_wf, qt_stay_c, b_stay_c, qm_c, lv_c, slot_next, vt_stay_nc, in_stay_nc_wf, qt_stay_nc, b_stay_nc, qm_nc, lv_nc, vt_renter, vt_buy_c, vt_buy_nc, in_renter_wf, in_buy_c_wf, in_buy_nc_wf, qt_renter, qt_buy_c, qt_buy_nc, v_owner_c_wf, v_owner_nc_wf, v_nonowner_wf, slot, t_index_prime, g_index_nc, dP_C, dP_NC, vt_stay_c_wf, vt_stay_nc_wf, c_c, c_nc, b_renter, vt_renter_wf)

            #####################################################################
            # (2) Mortgage price at (t,j): pooling over K and G at age j
            #####################################################################
            if j < par.iNj-1:
                pool_lender_values(par, grids, k_dim, lv_c, lv_nc, slot, j, share_sceptic, frac_switch, qm_c[t_index], qm_nc[t_index])

            #####################################################################
            # (3) Buyers (use q_{t,j}) and welfare bookkeeping, all types
            #####################################################################
            for type_index in prange(type_mat.shape[0]):
                _types_step3(type_index, type_mat, par, grids, j, dP_C, config, g_index_nc, dP_NC, vt_buy_c, qt_buy_c, vt_stay_c, c_c, b_stay_c, qm_c, in_stay_c_wf, vt_buy_nc, qt_buy_nc, vt_stay_nc, c_nc, b_stay_nc, qm_nc, in_stay_nc_wf, vt_buy_c_wf, vt_buy_nc_wf, t_index, slot, shape_stay, vt_renter, vt_renter_wf, qt_renter, t_index_start, vt_stay_c_wf, qt_stay_c, lv_c, vt_stay_nc_wf, qt_stay_nc, lv_nc, mortgage_size_C, mortgage_size_NC, v_nonowner_wf, v_owner_c_wf, v_owner_nc_wf, t_index_prime)


    return vt_stay_c, vt_stay_nc, vt_renter, b_stay_c, b_stay_nc, b_renter, v_owner_c_wf, v_owner_nc_wf, v_nonowner_wf, qm_c, qm_nc


@njit(parallel=PARALLEL)
def solve_ss(grids, par, dCoeff_C, dCoeff_NC, config):

    if config.sceptics==False:
        k_dim=1
    else:
        k_dim=grids.vK.size

    shape_stay                          = (1,par.iNj, k_dim, grids.vG.size, grids.vM.size, grids.vH.size, grids.vL.size, grids.vE.size)
    shape_buy                           = (1,par.iNj, k_dim, grids.vG.size, grids.vX.size,  grids.vE.size)
    shape_rent                          = (1,par.iNj, k_dim, grids.vG.size, grids.vX.size, grids.vE.size)
    shape_qm                            = (1, par.iNj, grids.vB.size, grids.vH.size, grids.vL.size, grids.vE.size)
    shape_lv                            = (1, par.iNj, k_dim, grids.vG.size, grids.vB.size, grids.vH.size, grids.vL.size, grids.vE.size)


    vt_buy_c        = np.zeros(shape_buy, dtype = np.float64)
    vt_buy_c_wf      = np.zeros(shape_buy, dtype = np.float64)
    qt_buy_nc       = np.zeros(shape_buy, dtype = np.float64)
    vt_stay_c       = np.zeros(shape_stay, dtype = np.float64)
    vt_stay_c_wf    = np.zeros(shape_stay, dtype = np.float64)
    qt_stay_c       = np.zeros(shape_stay, dtype = np.float64)
    vt_buy_nc       = np.zeros(shape_buy, dtype = np.float64)
    vt_buy_nc_wf    = np.zeros(shape_buy, dtype = np.float64)
    qt_buy_c        = np.zeros(shape_buy, dtype = np.float64)
    vt_stay_nc      = np.zeros(shape_stay, dtype = np.float64)
    vt_stay_nc_wf   = np.zeros(shape_stay, dtype = np.float64)
    qt_stay_nc      = np.zeros(shape_stay, dtype = np.float64)
    vt_renter  = np.zeros(shape_rent, dtype = np.float64)
    vt_renter_wf= np.zeros(shape_rent, dtype = np.float64)
    qt_renter  = np.zeros(shape_rent, dtype = np.float64)
    b_stay_c        = np.zeros(shape_stay, dtype = np.float64)
    b_stay_nc       = np.zeros(shape_stay, dtype = np.float64)
    b_renter        = np.zeros(shape_rent, dtype = np.float64)
    c_c             = np.zeros(shape_stay, dtype = np.float64)
    c_nc            = np.zeros(shape_stay, dtype = np.float64)
    qm_c            = np.ones(shape_qm, dtype = np.float64)*(1-par.dZeta)
    qm_nc           = np.ones(shape_qm, dtype = np.float64)*(1-par.dZeta)
    lv_c            = np.ones(shape_lv, dtype = np.float64)
    lv_nc           = np.ones(shape_lv, dtype = np.float64)

    #Matrix of expected welfare on a savings (before interest) grid
    v_owner_c_wf=np.zeros(shape_stay, dtype = np.float64)
    v_owner_nc_wf=np.zeros(shape_stay, dtype = np.float64)
    v_nonowner_wf=np.zeros(shape_rent, dtype = np.float64)

    mortgage_size_C = np.zeros((grids.vH.size,grids.vL.size), dtype = np.float64)
    mortgage_size_NC= np.zeros((grids.vH.size,grids.vL.size), dtype = np.float64)
    dP_C= dCoeff_C
    dP_NC = dCoeff_NC
    dP_C_prime = dCoeff_C
    dP_NC_prime = dCoeff_NC
    for h_index in range(grids.vH.size):
        for l_index in range(grids.vL.size):
            mortgage_size_C[h_index,l_index]=grids.vL[l_index]*grids.vH[h_index]*dP_C
            mortgage_size_NC[h_index,l_index]=grids.vL[l_index]*grids.vH[h_index]*dP_NC

    g_index_nc=int((grids.vG.size-1)/2)
    assert grids.vG[g_index_nc]==1

    if config.initial:
        dPi_L=grids.vPi_L[0]
        t_index=0
    else:
        dPi_L=grids.vPi_L[-1]
        t_index=grids.vTime.size-1

    if config.building_rest == True:
        arrival_rate_discount = (1-par.dMitigation_rate)
    else:
        arrival_rate_discount = 1

    if config.initial == True:
        dPi_S=par.vPi_S_median[0]
    else:
        dPi_S = par.vPi_S_median[-1]*arrival_rate_discount

    # Steady state: no belief switching (as in the stationary simulation), pooling with the date's share
    share_sceptic = 0.
    if k_dim==2:
        share_sceptic = grids.mTypes[t_index,1]
    frac_switch = 0.

    type_mat = misc.DoubleGrid(np.linspace(0, k_dim-1, k_dim), np.linspace(0, grids.vG.size-1, grids.vG.size))

    for j in range(par.iNj-1, -1, -1):
        # (1) continuation, lender values, stayers, renters
        for type_index in prange(type_mat.shape[0]):
            _types_step1_ss(type_index, type_mat, j, par, grids, dPi_S, dPi_L, dP_C_prime, mortgage_size_C, config, dP_NC_prime, mortgage_size_NC, shape_stay, vt_stay_c, vt_stay_c_wf, qt_stay_c, b_stay_c, qm_c, lv_c, vt_stay_nc, vt_stay_nc_wf, qt_stay_nc, b_stay_nc, qm_nc, lv_nc, vt_renter, vt_buy_c, vt_buy_nc, vt_renter_wf, vt_buy_c_wf, vt_buy_nc_wf, qt_renter, qt_buy_c, qt_buy_nc, v_owner_c_wf, v_owner_nc_wf, v_nonowner_wf, g_index_nc, t_index, dP_C, dP_NC, c_c, c_nc, b_renter)

        # (2)  pooling over K and G at age j
        if j < par.iNj-1:
            pool_lender_values(par, grids, k_dim, lv_c, lv_nc, 0, j, share_sceptic, frac_switch, qm_c[0], qm_nc[0])

        # (3) buyers and welfare bookkeeping
        for type_index in prange(type_mat.shape[0]):
            _types_step3_ss(type_index, type_mat, par, grids, j, dP_C, config, g_index_nc, dP_NC, vt_buy_c, qt_buy_c, vt_buy_c_wf, vt_stay_c, c_c, b_stay_c, qm_c, vt_stay_c_wf, vt_buy_nc, qt_buy_nc, vt_buy_nc_wf, vt_stay_nc, c_nc, b_stay_nc, qm_nc, vt_stay_nc_wf, shape_stay, vt_renter, vt_renter_wf, qt_renter, v_nonowner_wf)


    return vt_stay_c, vt_stay_nc, vt_renter, b_stay_c, b_stay_nc, b_renter, v_owner_c_wf, v_owner_nc_wf, v_nonowner_wf, qm_c, qm_nc



###########################################################
### Per-type work of one (date, age), run in parallel over types by solve / solve_ss
###########################################################

@njit
def _types_step1(type_index, type_mat, j, par, grids, dPi_S, dPi_L, dP_C_prime, mortgage_size_C, config, dP_NC_prime, mortgage_size_NC, shape_stay, t_index, vt_stay_c, in_stay_c_wf, qt_stay_c, b_stay_c, qm_c, lv_c, slot_next, vt_stay_nc, in_stay_nc_wf, qt_stay_nc, b_stay_nc, qm_nc, lv_nc, vt_renter, vt_buy_c, vt_buy_nc, in_renter_wf, in_buy_c_wf, in_buy_nc_wf, qt_renter, qt_buy_c, qt_buy_nc, v_owner_c_wf, v_owner_nc_wf, v_nonowner_wf, slot, t_index_prime, g_index_nc, dP_C, dP_NC, vt_stay_c_wf, vt_stay_nc_wf, c_c, c_nc, b_renter, vt_renter_wf):
    k_index = int(type_mat[type_index, 0])
    g_index = int(type_mat[type_index, 1])
    if j == par.iNj-1:
        w_c_last,q_c_last, w_c_wf_last = continuation_value_epsilons.solve_last_period_owners_C(par, grids,  dPi_S, dPi_L, k_index, dP_C_prime,mortgage_size_C, config)
        w_nc_last,q_nc_last, w_nc_wf_last  = continuation_value_epsilons.solve_last_period_owners_NC(par, grids,  k_index,  dP_NC_prime,mortgage_size_NC, config)
        w_renter_last,q_renter_last, w_renter_wf_last = continuation_value_epsilons.solve_last_period_renters(par, grids)
    else:
        coastal_stayer_inputs=precompute_coastal_stayer_inputs(shape_stay,j, k_index, g_index, t_index,
            vt_stay_c,in_stay_c_wf,qt_stay_c, b_stay_c, qm_c, lv_c, slot_next)
        noncoastal_stayer_inputs=precompute_noncoastal_stayer_inputs(shape_stay,j, k_index, g_index, t_index,
            vt_stay_nc,in_stay_nc_wf,qt_stay_nc, b_stay_nc, qm_nc, lv_nc, slot_next)
        mover_inputs=precompute_mover_inputs(shape_stay, j, k_index, g_index, t_index,
            vt_renter, vt_buy_c, vt_buy_nc,
            in_renter_wf, in_buy_c_wf, in_buy_nc_wf,
            qt_renter, qt_buy_c, qt_buy_nc)

        w_c_vE,q_c_vE, w_c_wf_vE, v_owner_c_wf_new, lv_c[slot,j,k_index,g_index] = continuation_value_epsilons.solve_owners_C(par, grids,  j+1, k_index,  dPi_S, dPi_L, coastal_stayer_inputs,mover_inputs,dP_C_prime, mortgage_size_C, config)
        w_nc_vE,q_nc_vE, w_nc_wf_vE,  v_owner_nc_wf_new, lv_nc[slot,j,k_index,g_index] = continuation_value_epsilons.solve_owners_NC(par, grids,  j+1, k_index, noncoastal_stayer_inputs,mover_inputs, dP_NC_prime,  mortgage_size_NC, config)
        w_renter_vE,q_renter_vE, w_renter_wf_vE, v_nonowner_wf_new = continuation_value_epsilons.solve_renters(par, grids,  j+1, k_index, mover_inputs, config)
        if config.welfare:
            v_owner_c_wf[t_index_prime,j+1, k_index,g_index,:,:,:,:] = v_owner_c_wf_new
            v_owner_nc_wf[t_index_prime,j+1, k_index,g_index,:,:,:,:] = v_owner_nc_wf_new
            v_nonowner_wf[t_index_prime,j+1, k_index,g_index,:,:] = v_nonowner_wf_new

    for e_index in range(grids.vE.size):
        if j == par.iNj-1:
            w_c,q_c, w_c_wf = w_c_last[:,:,:],q_c_last[:,:,:], w_c_wf_last[:,:,:]
            w_nc,q_nc, w_nc_wf  = w_nc_last[:,:,:],q_nc_last[:,:,:], w_nc_wf_last[:,:,:]
            w_renter,q_renter, w_renter_wf= w_renter_last[:],q_renter_last[:], w_renter_wf_last[:]
        else:
            w_c,q_c, w_c_wf = w_c_vE[:, :, :, e_index],q_c_vE[:, :, :, e_index], w_c_wf_vE[:, :, :, e_index]
            w_nc,q_nc, w_nc_wf = w_nc_vE[:, :, :, e_index],q_nc_vE[:, :, :, e_index], w_nc_wf_vE[:, :, :, e_index]
            w_renter,q_renter, w_renter_wf = w_renter_vE[:, e_index],q_renter_vE[:, e_index], w_renter_wf_vE[:, e_index]
        vt_stay_c[t_index,j, k_index,g_index,:,:,:,e_index], c_c[slot,j, k_index,g_index,:,:,:,e_index], qt_stay_c[slot,j, k_index,g_index,:,:,:,e_index], b_stay_c[t_index,j, k_index,g_index,:,:,:,e_index], vt_stay_c_wf_new = stayer_problem.solve(par, grids, j, k_index, g_index, w_c, q_c, w_c_wf, config)
        vt_stay_nc[t_index,j, k_index,g_index,:,:,:,e_index], c_nc[slot,j, k_index,g_index,:,:,:,e_index],  qt_stay_nc[slot,j, k_index,g_index,:,:,:,e_index], b_stay_nc[t_index,j, k_index,g_index,:,:,:,e_index], vt_stay_nc_wf_new = stayer_problem.solve(par, grids, j, k_index, g_index_nc, w_nc,q_nc, w_nc_wf, config)

        vt_renter[t_index,j, k_index,g_index,:,e_index],qt_renter[t_index,j, k_index,g_index,:,e_index], b_renter[t_index,j, k_index,g_index,:,e_index], vt_renter_wf_new = stayer_problem_renter.solve(par, grids, j, k_index, g_index, t_index, dP_C,dP_NC, dP_C_prime,dP_NC_prime, w_renter,q_renter, w_renter_wf, config)

        vt_stay_c_wf[slot,j, k_index,g_index,:,:,:,e_index] = vt_stay_c_wf_new
        vt_stay_nc_wf[slot,j, k_index,g_index,:,:,:,e_index] = vt_stay_nc_wf_new
        if config.welfare:
            vt_renter_wf[t_index,j, k_index,g_index,:,e_index] = vt_renter_wf_new


@njit
def _types_step3(type_index, type_mat, par, grids, j, dP_C, config, g_index_nc, dP_NC, vt_buy_c, qt_buy_c, vt_stay_c, c_c, b_stay_c, qm_c, in_stay_c_wf, vt_buy_nc, qt_buy_nc, vt_stay_nc, c_nc, b_stay_nc, qm_nc, in_stay_nc_wf, vt_buy_c_wf, vt_buy_nc_wf, t_index, slot, shape_stay, vt_renter, vt_renter_wf, qt_renter, t_index_start, vt_stay_c_wf, qt_stay_c, lv_c, vt_stay_nc_wf, qt_stay_nc, lv_nc, mortgage_size_C, mortgage_size_NC, v_nonowner_wf, v_owner_c_wf, v_owner_nc_wf, t_index_prime):
    k_index = int(type_mat[type_index, 0])
    g_index = int(type_mat[type_index, 1])
    for e_index in range(grids.vE.size):
        vt_buy_c[t_index,j, k_index,g_index,:,e_index], qt_buy_c[t_index,j, k_index,g_index,:,e_index], vt_buy_c_wf_new = buyer_problem_epsilons.solve(par, grids, j, k_index,  g_index, e_index, dP_C, grids.mPTI_C[j,e_index], vt_stay_c[t_index,j, k_index,g_index,:,:,:,e_index], c_c[slot,j, k_index,g_index,:,:,:,e_index], b_stay_c[t_index,j, k_index,g_index,:,:,:,e_index], qm_c[t_index,j,:,:,:,e_index], in_stay_c_wf[slot,j, k_index,g_index,:,:,:,e_index], config)
        vt_buy_nc[t_index,j, k_index,g_index,:,e_index], qt_buy_nc[t_index,j, k_index,g_index,:,e_index], vt_buy_nc_wf_new = buyer_problem_epsilons.solve(par, grids, j, k_index, g_index_nc, e_index, dP_NC, grids.mPTI_NC[j,e_index], vt_stay_nc[t_index,j, k_index,g_index,:,:,:,e_index], c_nc[slot,j, k_index,g_index,:,:,:,e_index], b_stay_nc[t_index,j, k_index,g_index,:,:,:,e_index], qm_nc[t_index,j,:,:,:,e_index], in_stay_nc_wf[slot,j, k_index,g_index,:,:,:,e_index], config)
        if config.welfare:
            vt_buy_c_wf[t_index,j, k_index,g_index,:,e_index] = vt_buy_c_wf_new
            vt_buy_nc_wf[t_index,j, k_index,g_index,:,e_index] = vt_buy_nc_wf_new

    #We need an additional block of code because the trick of exploiting already computed continuation values doesn't work when t==0 or j==0
    if config.welfare==True:
        if j==0:
            mover_inputs=precompute_mover_inputs(shape_stay, j-1, k_index, g_index, t_index,
                vt_renter, vt_buy_c, vt_buy_nc,
                vt_renter_wf, vt_buy_c_wf, vt_buy_nc_wf,
                qt_renter, qt_buy_c, qt_buy_nc)

            _,_, _, v_nonowner_wf[t_index_prime,j, k_index,g_index,:,:] = continuation_value_epsilons.solve_renters(par, grids,  j, k_index, mover_inputs, config)
        if t_index==t_index_start and j<par.iNj-1:
            dPi_S_0=grids.vPi_S_median[t_index]
            dPi_L_0=grids.vPi_L[t_index]
            # inputs dated t (t_index-1 -> t), lender values of date t (current slot)
            coastal_stayer_inputs=precompute_coastal_stayer_inputs(shape_stay,j, k_index, g_index, t_index-1,
                vt_stay_c,vt_stay_c_wf,qt_stay_c, b_stay_c, qm_c, lv_c, slot)
            noncoastal_stayer_inputs=precompute_noncoastal_stayer_inputs(shape_stay,j, k_index, g_index, t_index-1,
                vt_stay_nc,vt_stay_nc_wf,qt_stay_nc, b_stay_nc, qm_nc, lv_nc, slot)
            mover_inputs=precompute_mover_inputs(shape_stay, j, k_index, g_index, t_index-1,
                vt_renter, vt_buy_c, vt_buy_nc,
                vt_renter_wf, vt_buy_c_wf, vt_buy_nc_wf,
                qt_renter, qt_buy_c, qt_buy_nc)

            _,_, _, v_owner_c_wf[t_index,j+1, k_index,g_index,:,:,:,:], _ = continuation_value_epsilons.solve_owners_C(par, grids,  j+1, k_index,  dPi_S_0, dPi_L_0, coastal_stayer_inputs,mover_inputs,dP_C, mortgage_size_C,config)
            _,_, _,  v_owner_nc_wf[t_index,j+1, k_index,g_index,:,:,:,:], _  = continuation_value_epsilons.solve_owners_NC(par, grids,  j+1, k_index, noncoastal_stayer_inputs,mover_inputs, dP_NC,  mortgage_size_NC, config)
            _,_, _, v_nonowner_wf[t_index,j+1, k_index,g_index,:,:] = continuation_value_epsilons.solve_renters(par, grids,  j+1, k_index, mover_inputs, config)
        if t_index==t_index_start and j==0:
            mover_inputs=precompute_mover_inputs(shape_stay, j-1, k_index, g_index, t_index-1,
                vt_renter, vt_buy_c, vt_buy_nc,
                vt_renter_wf, vt_buy_c_wf, vt_buy_nc_wf,
                qt_renter, qt_buy_c, qt_buy_nc)

            _,_, _, v_nonowner_wf[t_index,j, k_index,g_index,:,:] = continuation_value_epsilons.solve_renters(par, grids,  j, k_index, mover_inputs, config)


@njit
def _types_step1_ss(type_index, type_mat, j, par, grids, dPi_S, dPi_L, dP_C_prime, mortgage_size_C, config, dP_NC_prime, mortgage_size_NC, shape_stay, vt_stay_c, vt_stay_c_wf, qt_stay_c, b_stay_c, qm_c, lv_c, vt_stay_nc, vt_stay_nc_wf, qt_stay_nc, b_stay_nc, qm_nc, lv_nc, vt_renter, vt_buy_c, vt_buy_nc, vt_renter_wf, vt_buy_c_wf, vt_buy_nc_wf, qt_renter, qt_buy_c, qt_buy_nc, v_owner_c_wf, v_owner_nc_wf, v_nonowner_wf, g_index_nc, t_index, dP_C, dP_NC, c_c, c_nc, b_renter):
    k_index = int(type_mat[type_index, 0])
    g_index = int(type_mat[type_index, 1])
    if j == par.iNj-1:
        w_c_last,q_c_last, w_c_wf_last = continuation_value_epsilons.solve_last_period_owners_C(par, grids,  dPi_S, dPi_L, k_index, dP_C_prime,mortgage_size_C, config)
        w_nc_last,q_nc_last, w_nc_wf_last  = continuation_value_epsilons.solve_last_period_owners_NC(par, grids,  k_index, dP_NC_prime,mortgage_size_NC, config)
        w_renter_last,q_renter_last, w_renter_wf_last = continuation_value_epsilons.solve_last_period_renters(par, grids)
    else:
        coastal_stayer_inputs=precompute_coastal_stayer_inputs(shape_stay,j, k_index, g_index, 0,
            vt_stay_c,vt_stay_c_wf,qt_stay_c, b_stay_c, qm_c, lv_c, 0)
        noncoastal_stayer_inputs=precompute_noncoastal_stayer_inputs(shape_stay,j, k_index, g_index, 0,
            vt_stay_nc,vt_stay_nc_wf,qt_stay_nc, b_stay_nc, qm_nc, lv_nc, 0)
        mover_inputs=precompute_mover_inputs(shape_stay, j, k_index, g_index, 0,
            vt_renter, vt_buy_c, vt_buy_nc,
            vt_renter_wf, vt_buy_c_wf, vt_buy_nc_wf,
            qt_renter, qt_buy_c, qt_buy_nc)

        w_c_vE,q_c_vE, w_c_wf_vE, v_owner_c_wf[0,j+1, k_index,g_index,:,:,:,:], lv_c[0,j,k_index,g_index] = continuation_value_epsilons.solve_owners_C(par, grids,  j+1, k_index, dPi_S,dPi_L, coastal_stayer_inputs,mover_inputs, dP_C_prime, mortgage_size_C, config)
        w_nc_vE,q_nc_vE, w_nc_wf_vE, v_owner_nc_wf[0,j+1, k_index,g_index,:,:,:,:], lv_nc[0,j,k_index,g_index] = continuation_value_epsilons.solve_owners_NC(par, grids,  j+1, k_index, noncoastal_stayer_inputs,mover_inputs, dP_NC_prime,  mortgage_size_NC, config)
        w_renter_vE,q_renter_vE, w_renter_wf_vE, v_nonowner_wf[0,j+1, k_index,g_index,:,:] = continuation_value_epsilons.solve_renters(par, grids,  j+1, k_index, mover_inputs, config)

    for e_index in range(grids.vE.size):
        if j == par.iNj-1:
            w_c,q_c, w_c_wf = w_c_last[:,:,:],q_c_last[:,:,:], w_c_wf_last[:,:,:]
            w_nc,q_nc, w_nc_wf  = w_nc_last[:,:,:],q_nc_last[:,:,:], w_nc_wf_last[:,:,:]
            w_renter,q_renter, w_renter_wf= w_renter_last[:],q_renter_last[:], w_renter_wf_last[:]
        else:
            w_c,q_c, w_c_wf= w_c_vE[:, :, :, e_index],q_c_vE[:, :, :, e_index], w_c_wf_vE[:, :, :, e_index]
            w_nc,q_nc, w_nc_wf = w_nc_vE[:, :, :, e_index],q_nc_vE[:, :, :, e_index], w_nc_wf_vE[:, :, :, e_index]
            w_renter,q_renter, w_renter_wf = w_renter_vE[:, e_index],q_renter_vE[:, e_index], w_renter_wf_vE[:, e_index]

        vt_stay_c[0,j, k_index,g_index,:,:,:,e_index], c_c[0,j, k_index,g_index,:,:,:,e_index], qt_stay_c[0,j, k_index,g_index,:,:,:,e_index], b_stay_c[0,j, k_index,g_index,:,:,:,e_index], vt_stay_c_wf[0,j, k_index,g_index,:,:,:,e_index] = stayer_problem.solve(par, grids, j, k_index, g_index, w_c, q_c, w_c_wf, config)
        vt_stay_nc[0,j, k_index,g_index,:,:,:,e_index], c_nc[0,j, k_index,g_index,:,:,:,e_index],  qt_stay_nc[0,j, k_index,g_index,:,:,:,e_index], b_stay_nc[0,j, k_index,g_index,:,:,:,e_index], vt_stay_nc_wf[0,j, k_index,g_index,:,:,:,e_index] = stayer_problem.solve(par, grids, j, k_index, g_index_nc, w_nc,q_nc, w_nc_wf, config)
        vt_renter[0,j, k_index,g_index,:,e_index],qt_renter[0,j, k_index,g_index,:,e_index], b_renter[0,j, k_index,g_index,:,e_index], vt_renter_wf[0,j, k_index,g_index,:,e_index] = stayer_problem_renter.solve(par, grids, j, k_index, g_index, t_index, dP_C,dP_NC, dP_C,dP_NC, w_renter,q_renter, w_renter_wf, config)


@njit
def _types_step3_ss(type_index, type_mat, par, grids, j, dP_C, config, g_index_nc, dP_NC, vt_buy_c, qt_buy_c, vt_buy_c_wf, vt_stay_c, c_c, b_stay_c, qm_c, vt_stay_c_wf, vt_buy_nc, qt_buy_nc, vt_buy_nc_wf, vt_stay_nc, c_nc, b_stay_nc, qm_nc, vt_stay_nc_wf, shape_stay, vt_renter, vt_renter_wf, qt_renter, v_nonowner_wf):
    k_index = int(type_mat[type_index, 0])
    g_index = int(type_mat[type_index, 1])
    for e_index in range(grids.vE.size):
        vt_buy_c[0,j, k_index,g_index,:,e_index], qt_buy_c[0,j, k_index,g_index,:,e_index], vt_buy_c_wf[0,j, k_index,g_index,:,e_index] = buyer_problem_epsilons.solve(par, grids, j, k_index,  g_index, e_index, dP_C, grids.mPTI_C[j,e_index], vt_stay_c[0,j, k_index,g_index,:,:,:,e_index], c_c[0,j, k_index,g_index,:,:,:,e_index], b_stay_c[0,j, k_index,g_index,:,:,:,e_index], qm_c[0,j,:,:,:,e_index], vt_stay_c_wf[0,j, k_index,g_index,:,:,:,e_index], config)
        vt_buy_nc[0,j, k_index,g_index,:,e_index], qt_buy_nc[0,j, k_index,g_index,:,e_index], vt_buy_nc_wf[0,j, k_index,g_index,:,e_index] = buyer_problem_epsilons.solve(par, grids, j, k_index, g_index_nc, e_index, dP_NC, grids.mPTI_NC[j,e_index], vt_stay_nc[0,j, k_index,g_index,:,:,:,e_index], c_nc[0,j, k_index,g_index,:,:,:,e_index], b_stay_nc[0,j, k_index,g_index,:,:,:,e_index], qm_nc[0,j,:,:,:,e_index], vt_stay_nc_wf[0,j, k_index,g_index,:,:,:,e_index], config)

    #We need an additional block of code because the trick of exploiting already computed continuation values doesn't work when t==0 or j==0
    if config.welfare==True:
        if j==0:
            mover_inputs=precompute_mover_inputs(shape_stay, j-1, k_index, g_index, 0,
                vt_renter, vt_buy_c, vt_buy_nc,
                vt_renter_wf, vt_buy_c_wf, vt_buy_nc_wf,
                qt_renter, qt_buy_c, qt_buy_nc)

            _,_, _, v_nonowner_wf[0,j, k_index,g_index,:,:] = continuation_value_epsilons.solve_renters(par, grids,  j, k_index, mover_inputs, config)


@njit
def precompute_coastal_stayer_inputs(shape_stay,j_index, k_index, g_index, t_index,
    vt_stay_c,vt_stay_c_wf,qt_stay_c, b_stay_c, qm_c, lv_c, lv_slot):
    # lv_slot: slot of the rolling arrays (lender values, qt_stay, vt_stay_wf) holding the date used here (t+1, or t in the terminal date)

    if t_index==shape_stay[0]-1:
        t_next = t_index
    else:
        t_next = t_index+1
    coastal_stayer_inputs = {
        'vt_stay_c_input': vt_stay_c[t_next,j_index+1, k_index, g_index, :,:, :, : ],
        'vt_stay_c_input_wf': vt_stay_c_wf[lv_slot,j_index+1, k_index, g_index, :,:, :, :],
        'qt_stay_c_input': qt_stay_c[lv_slot,j_index+1, k_index, g_index, :,:, :, :],
        'b_stay_c_input': b_stay_c[t_next,j_index+1, k_index, g_index, :,:, :, :],
        'qm_c_input': qm_c[t_next,j_index+1, :,:, :, :],
        'lv_c_input': lv_c[lv_slot,j_index+1, k_index, g_index, :,:, :, :],
         }

    return coastal_stayer_inputs

@njit
def precompute_noncoastal_stayer_inputs(shape_stay,j_index, k_index, g_index, t_index,
    vt_stay_nc,vt_stay_nc_wf,qt_stay_nc, b_stay_nc, qm_nc, lv_nc, lv_slot):

    if t_index==shape_stay[0]-1:
        t_next = t_index
    else:
        t_next = t_index+1
    noncoastal_stayer_inputs = {
        'vt_stay_nc_input': vt_stay_nc[t_next,j_index+1, k_index, g_index, :,:, :, :],
        'vt_stay_nc_input_wf': vt_stay_nc_wf[lv_slot,j_index+1, k_index, g_index,:,:, :, :],
        'qt_stay_nc_input': qt_stay_nc[lv_slot,j_index+1, k_index, g_index, :,:, :, :],
        'b_stay_nc_input': b_stay_nc[t_next,j_index+1, k_index, g_index, :,:, :, :],
        'qm_nc_input': qm_nc[t_next,j_index+1, :,:, :, :],
        'lv_nc_input': lv_nc[lv_slot,j_index+1, k_index, g_index, :,:, :, :],
         }
    return noncoastal_stayer_inputs


@njit
def precompute_mover_inputs(shape_stay,j_index, k_index, g_index,t_index,
    vt_renter, vt_buy_c, vt_buy_nc,
    vt_renter_wf, vt_buy_c_wf, vt_buy_nc_wf,
    qt_renter, qt_buy_c, qt_buy_nc):

    if t_index==shape_stay[0]-1:
        mover_inputs = {
            'vt_renter_input': vt_renter[t_index,j_index+1, k_index, g_index,  :, : ],
            'vt_buy_c_input': vt_buy_c[t_index,j_index+1, k_index, g_index, :, :],
            'vt_buy_nc_input': vt_buy_nc[t_index,j_index+1, k_index, g_index,  :, : ],
            'vt_renter_input_wf': vt_renter_wf[t_index,j_index+1, k_index, g_index,  :, :],
            'vt_buy_c_input_wf': vt_buy_c_wf[t_index,j_index+1, k_index, g_index,  :, : ],
            'vt_buy_nc_input_wf': vt_buy_nc_wf[t_index,j_index+1, k_index, g_index,  :, :],
            'qt_renter_input': qt_renter[t_index,j_index+1, k_index, g_index,  :, :],
            'qt_buy_c_input': qt_buy_c[t_index,j_index+1, k_index, g_index,  :, :],
            'qt_buy_nc_input': qt_buy_nc[t_index,j_index+1, k_index, g_index,  :, : ]
        }
    else:
        mover_inputs = {
            'vt_renter_input': vt_renter[t_index+1,j_index+1, k_index, g_index,  :, : ],
            'vt_buy_c_input': vt_buy_c[t_index+1,j_index+1, k_index, g_index,  :, :],
            'vt_buy_nc_input': vt_buy_nc[t_index+1,j_index+1, k_index, g_index, :, : ],
            'vt_renter_input_wf': vt_renter_wf[t_index+1,j_index+1, k_index, g_index, :, :],
            'vt_buy_c_input_wf': vt_buy_c_wf[t_index+1,j_index+1, k_index, g_index, :, : ],
            'vt_buy_nc_input_wf': vt_buy_nc_wf[t_index+1,j_index+1, k_index, g_index, :, :],
            'qt_renter_input': qt_renter[t_index+1,j_index+1, k_index, g_index, :, :],
            'qt_buy_c_input': qt_buy_c[t_index+1,j_index+1, k_index, g_index, :, :],
            'qt_buy_nc_input': qt_buy_nc[t_index+1,j_index+1, k_index, g_index, :, : ]
        }

    return mover_inputs

@njit
def compute_p_left(grid, x, i_left):

    x_left = grid[i_left]
    x_right = grid[i_left + 1]
    p_left = (x_right - x) / (x_right - x_left)

    return p_left

@njit
def pool_lender_values(par, grids, k_dim, lv_c, lv_nc, slot, j,
                       share_sceptic, frac_switch, qm_c_t, qm_nc_t):
    """
    Lenders observe neither belief type K nor amenity type G.
    Pool lender values using exogenous belief shares and
    uniform amenity weights. Offered q is identical for all G,K.
    """

    n_g = grids.vG.size
    g_weight = 1.0 / n_g # this is fine as long as G stays on a uniform grid

    # (1) Account for belief switching within each G
    if k_dim == 2 and frac_switch > 0.0:
        for g_index in range(n_g):
            lv_c[slot,j,1,g_index] = (
                (1-frac_switch)*lv_c[slot,j,1,g_index]
                + frac_switch*lv_c[slot,j,0,g_index]
            )
            lv_nc[slot,j,1,g_index] = (
                (1-frac_switch)*lv_nc[slot,j,1,g_index]
                + frac_switch*lv_nc[slot,j,0,g_index]
            )

    # (2) Pool across K and G (qm has no G dimension: every amenity type faces the same price)
    qm_c_t[j] = 0.0
    qm_nc_t[j] = 0.0

    for g_index in range(n_g):
        if k_dim == 2:
            qm_c_t[j] += g_weight * (
                share_sceptic*lv_c[slot,j,1,g_index]
                + (1-share_sceptic)*lv_c[slot,j,0,g_index]
            )
            qm_nc_t[j] += g_weight * (
                share_sceptic*lv_nc[slot,j,1,g_index]
                + (1-share_sceptic)*lv_nc[slot,j,0,g_index]
            )
        else:
            qm_c_t[j] += g_weight*lv_c[slot,j,0,g_index]
            qm_nc_t[j] += g_weight*lv_nc[slot,j,0,g_index]

    # (3) Apply proportional origination cost
    qm_c_t[j] *= (1-par.dZeta)
    qm_nc_t[j] *= (1-par.dZeta)
