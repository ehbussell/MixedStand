"""Parameters from Cobb (2012) for mixed stand model."""

import numpy as np


COBB_PARAMS = {
    # Infection rates
    'inf_tanoak_tanoak': np.array([0.33, 0.32, 0.30, 0.24]), # Cobb version
    'inf_bay_to_bay': 1.33,
    'inf_bay_to_tanoak': 1.46,
    'inf_tanoak_to_bay': 0.0, # In BM code from 2012 paper, there is no tanoak->bay infection

    # # Natural mortality rates
    'nat_mort_tanoak': np.array([0.0061, 0.0031, 0.0011, 0.032]),
    'nat_mort_bay': 0.02,
    'nat_mort_redwood': 0.02,

    # Pathogen induced mortality rates
    'inf_mort_tanoak': np.array([0.019, 0.022, 0.035, 0.14]), #Cobb Version

    # Tanoak age class transition rates
    'trans_tanoak': np.array([0.0521184, 0.0184760, 0.0147257]), # Parameters used in BM code from
                                                                 # 2012 paper. Unclear where from

    # Recovery rates
    'recov_tanoak': 0.01,
    'recov_bay': 0.1,

    # Seed recruitment rates
    # Any nan values are set in initialisation to ensure dynamic equilibrium at start
    'recruit_tanoak': np.array([np.nan, 0.007, 0.02, 0.073]),
    'recruit_bay': np.nan,
    'recruit_redwood': np.nan,

    'kernel_type': "nn",
    # Proportion of spores deposited within cell
    'spore_within': 1.0, # In BM code from 2012 paper spore deposition is incorrect
    'spore_between': 1.0,
    'num_nn': 4, # Either 4 nearest neighbours or 8, 4 used in 2012 paper, 8 used in SODDr

    # Tanoak reprouting probability
    'resprout_tanoak': 0.5,

    # Relative measure of per-capita space used by each species
    # Any nan values are set in initialisation to ensure dynamic equilibrium at start
    'space_tanoak': np.full(4, np.nan),
    'space_bay': 1,
    'space_redwood': 1,
}

CORRECTED_PARAMS = {
    # Infection rates
    'inf_tanoak_tanoak': np.array([0.33, 0.32, 0.30, 0.24]), # Cobb version
    'inf_bay_to_bay': 1.33,
    'inf_bay_to_tanoak': 1.46,
    'inf_tanoak_to_bay': 0.30,

    # Natural mortality rates
    'nat_mort_tanoak': np.array([0.0061, 0.0031, 0.0011, 0.032]),
    'nat_mort_bay': 0.02,
    'nat_mort_redwood': 0.02,

    # Pathogen induced mortality rates
    'inf_mort_tanoak': np.array([0.019, 0.022, 0.035, 0.14]), #Cobb Version

    # Tanoak age class transition rates
    'trans_tanoak': np.array([0.0521184, 0.0184760, 0.0147257]), # Parameters used in BM code from
                                                                 # 2012paper. Unclear where from

    # Recovery rates
    'recov_tanoak': 0.01,
    'recov_bay': 0.1,

    # Seed recruitment rates
    # Any nan values are set in initialisation to ensure dynamic equilibrium at start
    'recruit_tanoak': np.array([np.nan, 0.007, 0.02, 0.073]),
    'recruit_bay': np.nan,
    'recruit_redwood': np.nan,

    'kernel_type': 'exp',
    'spore_within': 0.5,
    'kernel_range': 10,
    'kernel_scale': 0.5,

    # Tanoak reprouting probability
    'resprout_tanoak': 0.5,

    # Relative measure of per-capita space used by each species
    # Any nan values are set in initialisation to ensure dynamic equilibrium at start
    'space_tanoak': np.full(4, np.nan),
    'space_bay': 1,
    'space_redwood': 1,
}


COBB_INIT_FIG4A = np.array([
    0.1293 * 0.2985, 0.0, 0.0,
    0.3278 * 0.2985, 0.0, 0.0,
    0.3812 * 0.2985, 0.0, 0.0,
    (1 - 0.1293 - 0.3278 - 0.3812) * 0.2985, 0.0, 0.0,
    1.5*0.077, 0.0,
    1.5*0.217])

COBB_PROP_FIG4A = np.array([0.40, 0.16, 0.44])

# This set uncertain as does not exactly reproduce Figure 4b
COBB_INIT_FIG4B = np.array([
    0.1293 * 0.7, 0.0, 0.0,
    0.3278 * 0.7, 0.0, 0.0,
    0.3812 * 0.7, 0.0, 0.0,
    (1 - 0.1293 - 0.3278 - 0.3812) * 0.7, 0.0, 0.0,
    0.0, 0.0,
    0.19])

COBB_INIT_FIG4C = np.array([
    0.1293 * 0.08, 0.0, 0.0,
    0.3278 * 0.08, 0.0, 0.0,
    0.3812 * 0.08, 0.0, 0.0,
    (1 - 0.1293 - 0.3278 - 0.3812) * 0.08, 0.0, 0.0,
    0.0, 0.0,
    0.69])
