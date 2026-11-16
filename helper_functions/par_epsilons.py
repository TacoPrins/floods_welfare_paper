
import numpy as np

time_increment=2
vPi_S_median = np.array([0.01076, 0.01091, 0.01105, 0.01120, 0.01134, 0.01149, 0.01164, 0.01178, 0.01193, 0.01208, 0.01222, 0.01237, 0.01252,
                         0.01266, 0.01281, 0.01295, 0.01310, 0.01325, 0.01339, 0.01354, 0.01369, 0.01383, 0.01398, 0.01413, 0.01427, 0.01442,
                         0.01456, 0.01471, 0.01486, 0.01500, 0.01515, 0.01530, 0.01544, 0.01559, 0.01574, 0.01588, 0.01603, 0.01618, 0.01632,
                         0.01647, 0.01661, 0.01676, 0.01691, 0.01705, 0.01720, 0.01735, 0.01749, 0.01764, 0.01779, 0.01793, 0.01808, 0.01822,
                         0.01822, 0.01822, 0.01822, 0.01822, 0.01822, 0.01822])
vPi_S_median=1-(1-vPi_S_median)**time_increment    


par_dict = {"time_increment": time_increment,
          "iNj": 30,
          "j_ret": 23,
          "dBeta": 9.42388105e-01**time_increment, 
          "dDelta": 1-(1-0.015)**time_increment, 
          "dPsi": 1.22747018e-02,
          "dDelta_deprec_rental": 1-(1-0.015)**time_increment,
          "dDelta_default": 0,
          "r": 1.03**time_increment-1, 
          "r_m_c": 1.04**time_increment-1, 
          'r_m_c_experiment': 1.06**time_increment-1, 
          "r_m_nc": 1.04**time_increment-1, 
          'vPi_S_median': vPi_S_median,
          "dKappa_sell": 0.07,
          "dKappa_buy": 0,
          "dXi_foreclosure": 0.8,
          "dNu": 4.59859461e+01,
          "dZeta": 0.01, 
          "dZeta_fixed": 1/26, 
          "lambda_pti":0.25,
          "max_ltv": 0.95,
          "damage_states": 3,
          "dLambda": 0.8,
          'wf_wedge': np.array([0.]),
          "dGamma": 1/1.25,
          "dSigma": 2,
          "b_bar":3.69412935e+00,
          "dPhi": 1.56281058e-01,
          "nonlingrid": 1,
          "nonlingrid_big": 1,  
          "iNb":70,
          "iBmin": 0, 
          "iBmax": 27,
          "dZ":0.8,
          "h_max": 5.15,
          "dXi_min":1-4.65839810e-02,          
          "dXi_max": 1+4.65839810e-02,
          "iXin": 5, 
          "alpha_0": 0.4,
          "dRho": 0.97, 
          "dSigmaeps": 0.20, 
          "dSigmaeps_trans": 0.05, 
          "iNumStates":5, #THIS NUMBER MUST BE UNEVEN
          "iNumTrans":3, 
          "iM":1,
          'vAgeEquiv':np.ones(50),
          'dNC_frac': 0.875,
          'dC_frac': 0.125,
          'dTheta': 1.5/2.5, 
          'dL': 0.311,
          'dOmega':5.47901061e-03,
          'dMitigation_rate': 0.55,
          'building_rest_cost': 0.14,
          'sd_income_initial': .3228617,
          'beta0_nowealth': -.9910767, #=Constant in logit regression nowealth=1 on log income in lowest age bracket
          'beta1_nowealth': -.3417672, #=Coeff on log income (in terms of sds away from the median) in logit regression nowealth=1 on log income in lowest age bracket
          'var_poswealth': 1.557037, #=Variance log wealth conditional on wealth>0 in lowest age bracket
          'beta0_age': 2.0017337, #=Constant of fitted line age --> mean log income 
          'beta1_age': .78795906  , #=Coeff of age for fitted line age --> mean log income 
          'beta2_age': -.02717023, #=Coeff of age^2 for fitted line age --> mean log income  
          'beta3_age': .00042535, #=Coeff of age^3 for fitted line age --> mean log income 
          'beta4_age':  -2.505727467e-06, #=Coeff of age^4 for fitted line age --> mean log income 
          'corr_poswealth_income': 0.1984, #=Correlation between log income (in terms of sds away from the median) and log wealth conditional on wealth>0 in lowest age bracket
          'tau_0': 0.75,
          'tau_1': 0.151,
          'building_rest': 0.5,
          'starting_year': 1998,
          'experiment_year': 2026,
          'order_polynomial': 4, #The order of the polynomial in the LoM for prices
          }