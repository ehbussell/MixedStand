"""Gemeral useful utilities."""

from enum import IntEnum
import logging
import numpy as np

class Species(IntEnum):
    """Host species."""
    TANOAK = 0
    BAY = 1
    REDWOOD = 2

def objective_integrand(time, state, control, params):
    """Integrand of objective function, including control costs and diversity costs."""

    integrand = 0.0

    div_cost = params.get('div_cost', 0.0)

    if div_cost != 0.0:
        state = np.sum(np.reshape(state, (-1, 15)), axis=0)

        props = np.divide(np.array([np.sum(state[0:6]), np.sum(state[6:12]),
                                    state[12] + state[13], state[14]]),
                          np.sum(state[0:15]), out=np.zeros(4),
                          where=(np.sum(state[0:15]) > 0.0))

        integrand += div_cost * np.sum(
            props * np.log(props, out=np.zeros_like(props), where=(props > 0.0)))

    integrand *= np.exp(- params.get('discount_rate', 0.0) * time)

    return integrand

def objective_payoff(end_time, state, params):
    """Payoff term in objective function - Healthy large tanoak"""

    state = np.sum(np.reshape(state, (-1, 15)), axis=0)

    payoff = - params.get('payoff_factor', 0.0) * np.exp(
        - params.get('discount_rate', 0.0) * end_time) * (
            state[6] + state[8] + state[9] + state[11])

    return payoff

def initialise_params(params, init_state=None, host_props=None):
    """Sets up initial conditions, space weights and recruitment rates to give dynamic equilibrium.

    Following Cobb (2012), we choose conditions and parameters that give dynamic equilibrium. In
    code from Cobb paper the initial state (numbers in each age class and empty space) are used to
    calculate the recruitment rate for the smallest tanoak class. The space weights for each class
    are found by assuming 1/4 of space is taken up by each age class.
    This leads to unrealistic values.

    Using the host proportions (i.e. tanoak, bay and redwood - no breakdown by age class), and by
    setting recruitment from smallest tanoak age class to zero, we can find initial conditions and
    empty space to give dynamic equilibrium. We find the space weights by assuming space taken is
    approximately proportional to basal area.

    If init state is provided (length 15*ncells array) then the Cobb method is used. If host props
    (p_tan, p_bay, p_red) is provided then our method is used. New params dict and init state are
    returned (disease free if new method used).

    Note values that are not NaN will not be overwritten apart from b11 in new method.
    """

    if init_state is None and host_props is None:
        raise RuntimeError("Must specify either state and proportions!")

    if init_state is not None and host_props is not None:
        raise RuntimeError("Cannot specify both state and proportions!")

    if init_state is not None:
        logging.info("Initialising using method from Cobb 2012 paper.")

        # Initialise rates s.t. if initial state was disease free, it is in dynamic equilibrium
        # Follows calculation in Cobb (2012)
        # First find number in each state if disease free
        ncells = len(init_state) / 15
        avg_state_init = np.sum(np.reshape(init_state, (-1, 15)), axis=0) / ncells
        avg_df_state_init = np.array(
            [np.sum(avg_state_init[3*i:3*i+3]) for i in range(4)] +
            [avg_state_init[12] + avg_state_init[13]] + [avg_state_init[14]])

        # Space weights:
        if np.all(avg_df_state_init[:4] > 0):
            if np.all(np.isnan(params['space_tanoak'])):
                params['space_tanoak'] = 0.25 * np.sum(
                    avg_df_state_init[:4]) / avg_df_state_init[:4]
        else:
            # No tanoak - set space weights to zero
            params['space_tanoak'] = np.repeat(0.0, 4)

        # Recruitment rates:
        # Any recruitment rates that are nan in parameters are chosen to give dynamic equilibrium
        # See online SI of Cobb (2012) for equations
        space_at_start = (1.0 - np.sum(params['space_tanoak'] * avg_df_state_init[:4]) -
                          params['space_bay'] * avg_df_state_init[4] -
                          params['space_redwood'] * avg_df_state_init[5])

        if np.isnan(params['recruit_bay']):
            params['recruit_bay'] = params['nat_mort_bay'] / space_at_start

        if np.isnan(params['recruit_redwood']):
            params['recruit_redwood'] = params['nat_mort_redwood'] / space_at_start

        if np.isnan(params['recruit_tanoak'][0]):
            A2 = params['trans_tanoak'][0] / (
                params['trans_tanoak'][1] + params['nat_mort_tanoak'][1])
            A3 = A2 * params['trans_tanoak'][1] / (
                params['trans_tanoak'][2] + params['nat_mort_tanoak'][2])
            A4 = A3 * params['trans_tanoak'][2] / params['nat_mort_tanoak'][3]

            params['recruit_tanoak'][0] = (
                (params['trans_tanoak'][0] + params['nat_mort_tanoak'][0]) /
                space_at_start - np.sum(params['recruit_tanoak'][1:] * np.array([A2, A3, A4])))

    if host_props is not None:
        logging.info("Initialising using new method.")

        # Set recruitment rate of smallest age class to zero
        params['recruit_tanoak'][0] = 0.0

        # First find empty space that gives dS11/dt = 0 when b11=0
        A2 = params['trans_tanoak'][0] / (
            params['trans_tanoak'][1] + params['nat_mort_tanoak'][1])
        A3 = A2 * params['trans_tanoak'][1] / (
            params['trans_tanoak'][2] + params['nat_mort_tanoak'][2])
        A4 = A3 * params['trans_tanoak'][2] / params['nat_mort_tanoak'][3]

        E0 = (params['trans_tanoak'][0] + params['nat_mort_tanoak'][0]) / (
            np.sum(params['recruit_tanoak'][1:] * np.array([A2, A3, A4])))

        # Bay recruitment rate:
        if np.isnan(params['recruit_bay']):
            params['recruit_bay'] = params['nat_mort_bay'] / E0

        # Redwood recruitment rate:
        if np.isnan(params['recruit_redwood']):
            params['recruit_redwood'] = params['nat_mort_redwood'] / E0

        # Now find age class numbers that give dynamic equilibrium
        S11 = host_props[0] * (1 - E0) / (1 + A2 + A3 + A4)
        S12 = S11 * params['trans_tanoak'][0] / (
            params['trans_tanoak'][1] + params['nat_mort_tanoak'][1])
        S13 = S12 * params['trans_tanoak'][1] / (
            params['trans_tanoak'][2] + params['nat_mort_tanoak'][2])
        S14 = S13 * params['trans_tanoak'][2] / params['nat_mort_tanoak'][3]

        S2 = host_props[1] * (1 - E0)
        S3 = host_props[2] * (1 - E0)

        init_state = np.array([
            S11, 0.0, 0.0, S12, 0.0, 0.0, S13, 0.0, 0.0, S14, 0.0, 0.0, S2, 0.0, S3])

        # Now find space weights
        # Approximately in proportion to basal area:
        weights = np.array([2.25, 36, 400, 2500])
        # Now ensure total empty space is E0 as already calculated
        weights *= (1 - S2 - S3 - E0) / np.sum(weights * np.array([S11, S12, S13, S14]))
        params['space_tanoak'] = weights

    return params, init_state
