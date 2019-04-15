"""Gemeral useful utilities."""

from enum import IntEnum
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
    cull_cost = params.get('cull_cost', 0.0)
    protect_cost = params.get('protect_cost', 0.0)

    if div_cost == cull_cost == protect_cost == 0:
        return integrand

    state = np.sum(np.reshape(state, (-1, 15)), axis=0)

    if div_cost != 0.0:
        props = np.divide(np.array([np.sum(state[0:6]), np.sum(state[6:12]),
                                    state[12] + state[13], state[14]]),
                          np.sum(state[0:15]), out=np.zeros(4),
                          where=(np.sum(state[0:15]) > 0.0))

        integrand += div_cost * np.sum(
            props * np.log(props, out=np.zeros_like(props), where=(props > 0.0)))

    if cull_cost != 0.0:
        integrand += (
            cull_cost * params.get('control_rate', 0.0) * (
                control[0] * (state[1] + state[4]) + control[1] * (state[7] + state[10]) +
                control[2] * state[13] + control[3] * np.sum(state[0:6]) +
                control[4] * np.sum(state[6:12]) + control[5] * (state[12] + state[13]) +
                control[6] * state[14]
            )
        )

    if protect_cost != 0.0:
        integrand += (
            protect_cost * params.get('control_rate', 0.0) *
            (control[7] * (state[0] + state[3]) + control[8] * (state[6] + state[9])))

    integrand *= np.exp(- params.get('discount_rate', 0.0) * time)

    return integrand

def objective_payoff(end_time, state, params):
    """Payoff term in objective function - Healthy large tanoak"""

    state = np.sum(np.reshape(state, (-1, 15)), axis=0)

    payoff = - params.get('payoff_factor', 0.0) * np.exp(
        - params.get('discount_rate', 0.0) * end_time) * (
            state[6] + state[8] + state[9] + state[11])

    return payoff
