// Function for the dynamics of the problem
// dy/dt = dynamics(y,u,z,p)

// The following are the input and output available variables 
// for the dynamics of your optimal control problem.

// Input :
// time : current time (t)
// normalized_time: t renormalized in [0,1]
// initial_time : time value on the first discretization point
// final_time : time value on the last discretization point
// dim_* is the dimension of next vector in the declaration
// state : vector of state variables
// control : vector of control variables
// algebraicvars : vector of algebraic variables
// optimvars : vector of optimization parameters
// constants : vector of constants

// Output :
// state_dynamics : vector giving the expression of the dynamic of each state variable.

// The functions of your problem have to be written in C++ code
// Remember that the vectors numbering in C++ starts from 0
// (ex: the first component of the vector state is state[0])

// Tdouble variables correspond to values that can change during optimization:
// states, controls, algebraic variables and optimization parameters.
// Values that remain constant during optimization use standard types (double, int, ...).

#include "header_dynamics"
#include "adolc/adolc.h"
#include "adolc/adouble.h"
#include <iostream>
{
	// Implement mixed stand infection dynamics

	// Define parameters
	double beta[] = {constants[0], constants[1], constants[2], constants[3], constants[4],
					 constants[5], constants[6]};
	double space[] = {constants[7], constants[8], constants[9], constants[10], constants[11],
					  constants[12]};
	double recruit[] = {constants[13], constants[14], constants[15], constants[16], constants[17],
						constants[18]};
	double nat_mort[] = {constants[19], constants[20], constants[21], constants[22], constants[23],
						 constants[24]};
	double inf_mort[] = {constants[25], constants[26], constants[27], constants[28]};
	double resprout = constants[29];
	double age_trans[] = {constants[30], constants[31], constants[32]};
	double recov[] = {constants[33], constants[34]};
	double primary_inf = constants[35];
	double rogue_rate = constants[36];
	double thin_rate = constants[37];
	double protect_rate = constants[38];
	double treat_eff = constants[39];
	double div_cost = constants[40];
	double control_cost = constants[41];
	double rogue_cost = constants[42];
	double thin_cost = constants[43];
	double rel_small_cost = constants[44];
	double protect_cost = constants[45];
	double discount_rate = constants[46];
	double vaccine_decay = constants[49];

	Tdouble inf_rate = 0.0;
	Tdouble empty_space = 1.0 - space[4] * (state[12] + state[13]) - space[5] * state[14];
	for (int i=0; i<4; i++){
		empty_space -= space[i] * (state[3*i] + state[3*i+1] + state[3*i+2]);
	}

	Tdouble tanoak = fabs(state[0] + state[1] + state[2] + state[3] + state[4] + state[5] + 
						  state[6] + state[7] + state[8] + state[9] + state[10] + state[11]);
	Tdouble bay = fabs(state[12] + state[13]);
	Tdouble red = fabs(state[14]);

	// Find total number of hosts
	Tdouble nhosts = tanoak + bay + red;

	Tdouble diversity_costs = log(
		pow(tanoak / nhosts, tanoak / nhosts) * pow(bay / nhosts, bay / nhosts) *
		pow(red / nhosts, red / nhosts));
	
	// Tdouble control_costs = 
	// 	(control[0] * rel_small_cost * rogue_rate * rogue_cost * (state[1] + state[4])) +
	// 	(control[1] * (state[7] + state[10]) + control[2] * (state[13])) * rogue_rate * rogue_cost +
	// 	(control[3] * rel_small_cost  * thin_rate * thin_cost * (
	// 		state[0] + state[1] + state[2] + state[3] + state[4] + state[5])) +
	// 	(control[4] * (state[6] + state[7] + state[8] + state[9] + state[10] + state[11]) +
	// 		control[5] * (state[12] + state[13]) + control[6] * state[14]) * thin_rate * thin_cost +
	// 	(control[7] * (state[0] + state[1] + state[2] + state[3] + state[4] + state[5]) +
	// 		control[8] * (state[6] + state[7] + state[8] + state[9] + state[10] + state[11])) *
	// 		protect_rate * protect_cost;

	Tdouble control_costs = (control[0] * rel_small_cost * rogue_rate * rogue_cost) +
		(control[1] + control[2]) * rogue_rate * rogue_cost +
		(control[3] * rel_small_cost  * thin_rate * thin_cost) +
		(control[4] + control[5] + control[6]) * thin_rate * thin_cost +
		(control[7] + control[8]) * protect_rate * protect_cost;
	
	// Objective function
	state_dynamics[15] = exp(- discount_rate * time) * 
		(div_cost * diversity_costs + control_cost * control_costs);

	// Dynamics
	// Initialise to zero
	for (int i=0; i<15; i++){
		state_dynamics[i] = 0.0;
	}

	// Tanoak dynamics
	for (int age=0; age<4; age++)	{
		// Recruitment
		state_dynamics[0] += recruit[age] * (state[3*age] + state[3*age+1] + state[3*age+2])
								* empty_space;
		
		// Mortality, recovery & resprouting
		state_dynamics[3*age] += - nat_mort[age] * state[3*age] + recov[0] * state[3*age+1]
									+ vaccine_decay * state[3*age+2];
		state_dynamics[3*age+1] += - (nat_mort[age] + inf_mort[age] + recov[0]) * state[3*age+1];
		state_dynamics[3*age+2] += - (nat_mort[age] + vaccine_decay) * state[3*age+2];
		state_dynamics[0] += inf_mort[age] * resprout * state[3*age+1];

		// Age transitions
		if (age < 3){
			for (int i=0; i<3; i++){
				state_dynamics[3*age+i] -= age_trans[age] * state[3*age+i];
				state_dynamics[3*(age+1)+i] += age_trans[age] * state[3*age+i];
			}
		}

		// Infection
		inf_rate = primary_inf +
			(beta[age] * (state[1] + state[4] + state[7] + state[10]) +
			beta[4] * state[13]);
		state_dynamics[3*age] -= state[3*age] * inf_rate;
		state_dynamics[3*age+1] += inf_rate * (state[3*age] + state[3*age+2] * treat_eff);
		state_dynamics[3*age+2] -= state[3*age+2] * inf_rate * treat_eff;

	}

	// Bay and Redwood dynamics
	// Recruitment
	state_dynamics[12] += recruit[4] * (state[12] + state[13]) * empty_space;
	state_dynamics[14] += recruit[5] * state[14] * empty_space;

	// Mortality & recovery
	state_dynamics[12] += - nat_mort[4] * state[12] + recov[1] * state[13];
	state_dynamics[13] += - (nat_mort[4] + recov[1]) * state[13];
	state_dynamics[14] += - nat_mort[5] * state[14];

	// Bay infection
	inf_rate = primary_inf * state[12] +
		state[12] * (beta[5] * (state[1] + state[4] + state[7] + state[10]) + beta[6] * state[13]);
	state_dynamics[12] -= inf_rate;
	state_dynamics[13] += inf_rate;

	// CONTROL
	// Roguing
	state_dynamics[1] -= control[0] * rogue_rate * state[1];
	state_dynamics[4] -= control[0] * rogue_rate * state[4];
	state_dynamics[7] -= control[1] * rogue_rate * state[7];
	state_dynamics[10] -= control[1] * rogue_rate * state[10];
	state_dynamics[13] -= control[2] * rogue_rate * state[13];

	// Thinning
	// Small Tanoak
	state_dynamics[0] -= control[3] * thin_rate * state[0];
	state_dynamics[1] -= control[3] * thin_rate * state[1];
	state_dynamics[2] -= control[3] * thin_rate * state[2];
	state_dynamics[3] -= control[3] * thin_rate * state[3];
	state_dynamics[4] -= control[3] * thin_rate * state[4];
	state_dynamics[5] -= control[3] * thin_rate * state[5];
	// Large Tanoak
	state_dynamics[6] -= control[4] * thin_rate * state[6];
	state_dynamics[7] -= control[4] * thin_rate * state[7];
	state_dynamics[8] -= control[4] * thin_rate * state[8];
	state_dynamics[9] -= control[4] * thin_rate * state[9];
	state_dynamics[10] -= control[4] * thin_rate * state[10];
	state_dynamics[11] -= control[4] * thin_rate * state[11];
	// Bay
	state_dynamics[12] -= control[5] * thin_rate * state[12];
	state_dynamics[13] -= control[5] * thin_rate * state[13];
	// Redwood
	state_dynamics[14] -= control[6] * thin_rate * state[14];

	// Phosphonite protectant
	state_dynamics[0] -= control[7] * protect_rate * state[0];
	state_dynamics[2] += control[7] * protect_rate * state[0];
	state_dynamics[3] -= control[7] * protect_rate * state[3];
	state_dynamics[5] += control[7] * protect_rate * state[3];
	state_dynamics[6] -= control[8] * protect_rate * state[6];
	state_dynamics[8] += control[8] * protect_rate * state[6];
	state_dynamics[9] -= control[8] * protect_rate * state[9];
	state_dynamics[11] += control[8] * protect_rate * state[9];

}
