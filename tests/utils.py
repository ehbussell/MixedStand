"""General utilities required by ultiple tests."""

import logging
import numpy as np

ZERO_PARAMS = {
    # Infection rates
    'inf_tanoak_tanoak': np.array([0.0, 0.0, 0.0, 0.0]),
    'inf_bay_to_bay': 0.0,
    'inf_bay_to_tanoak': 0.0,
    'inf_tanoak_to_bay': 0.0,

    # # Natural mortality rates
    'nat_mort_tanoak': np.array([0.0, 0.0, 0.0, 0.0]),
    'nat_mort_bay': 0.0,
    'nat_mort_redwood': 0.0,

    # Pathogen induced mortality rates
    'inf_mort_tanoak': np.array([0.0, 0.0, 0.0, 0.0]),

    # Tanoak age class transition rates
    'trans_tanoak': np.array([0.0, 0.0, 0.0]),

    # Recovery rates
    'recov_tanoak': 0.0,
    'recov_bay': 0.0,

    # Seed recruitment rates
    # Any nan values are set in initialisation to ensure dynamic equilibrium at start
    'recruit_tanoak': np.array([0.0, 0.0, 0.0, 0.0]),
    'recruit_bay': 0.0,
    'recruit_redwood': 0.0,

    # Proportion of spores deposited within cell
    'spore_within': 1.0,
    'spore_between': 0.0,
    'num_nn': 4,

    # Tanoak reprouting probability
    'resprout_tanoak': 0.0,

    # Relative measure of per-capita space used by each species
    # Any nan values are set in initialisation to ensure dynamic equilibrium at start
    'space_tanoak': np.full(4, 1.0),
    'space_bay': 1.0,
    'space_redwood': 1.0,
}

def sis_analytic(times, beta, mu, I0, N):
    """Analytic solution for SIS model: dI/dt = beta*I*(N-I) - mu * I"""

    logging.info("Calculating analytic SIS solution.")
    A = np.exp((beta*N - mu) * times)

    return (beta*N - mu) * I0 * A / (beta*N - mu + beta*I0*(A - 1.0))

def get_sis_params(beta, mu):
    """Construct parameters for non-spatial SIS model."""

    params = {
        # Infection rates
        'inf_tanoak_tanoak': np.array([beta, 0.0, 0.0, 0.0]),
        'inf_bay_to_bay': 0.0,
        'inf_bay_to_tanoak': 0.0,
        'inf_tanoak_to_bay': 0.0,

        # # Natural mortality rates
        'nat_mort_tanoak': np.array([0.0, 0.0, 0.0, 0.0]),
        'nat_mort_bay': 0.0,
        'nat_mort_redwood': 0.0,

        # Pathogen induced mortality rates
        'inf_mort_tanoak': np.array([0.0, 0.0, 0.0, 0.0]),

        # Tanoak age class transition rates
        'trans_tanoak': np.array([0.0, 0.0, 0.0]),

        # Recovery rates
        'recov_tanoak': mu,
        'recov_bay': 0.0,

        # Seed recruitment rates
        # Any nan values are set in initialisation to ensure dynamic equilibrium at start
        'recruit_tanoak': np.array([0.0, 0.0, 0.0, 0.0]),
        'recruit_bay': 0.0,
        'recruit_redwood': 0.0,

        # Proportion of spores deposited within cell
        'spore_within': 1.0,
        'spore_between': 0.0,
        'num_nn': 4,

        # Tanoak reprouting probability
        'resprout_tanoak': 0.0,

        # Relative measure of per-capita space used by each species
        # Any nan values are set in initialisation to ensure dynamic equilibrium at start
        'space_tanoak': np.full(4, 1.0),
        'space_bay': 1.0,
        'space_redwood': 1.0,
    }

    return params
