// Function for the path constraints of the problem
// a <= g(t,y,u,z,p) <= b

// The following are the input and output available variables 
// for the path constraints of your optimal control problem.

// Input :
// dim_path_constraints : number of path constraints
// time : current time (t)
// initial_time : time value on the first discretization point
// final_time : time value on the last discretization point
// dim_* is the dimension of next vector in the declaration
// state : vector of state variables
// control : vector of control variables
// algebraicvars : vector of algebraic variables
// optimvars : vector of optimization vector of optimization parameters
// constants : vector of constants

// Output :
// path_constraints : vector of path constraints expressions ("g" in the example above)

// The functions of your problem have to be written in C++ code
// Remember that the vectors numbering in C++ starts from 0
// (ex: the first component of the vector state is state[0])

// Tdouble variables correspond to values that can change during optimization:
// states, controls, algebraic variables and optimization parameters.
// Values that remain constant during optimization use standard types (double, int, ...).

#include "header_pathcond"

{
	double rogue_rate = constants[36];
	double thin_rate = constants[37];
	double protect_rate = constants[38];

	double rogue_cost = constants[42];
	double thin_cost = constants[43];
	double rel_small_cost = constants[44];
	double protect_cost = constants[45];

	int n_stages = constants[48];

	Tdouble small_tan = state[0] + state[1] + state[2] + state[3] + state[4] + state[5];
	Tdouble large_tan = state[6] + state[7] + state[8] + state[9] + state[10] + state[11];

	// Total control expenditure <= budget
	path_constraints[0] = 
		control[0] * rel_small_cost * rogue_rate * rogue_cost * (state[1] + state[4]) +
		(control[1] * (state[7] + state[10]) + control[2] * state[13]) * rogue_rate * rogue_cost +
		control[3] * rel_small_cost  * thin_rate * thin_cost * small_tan +
		(control[4] * large_tan + control[5] * (state[12] + state[13]) + control[6] * state[14]) * thin_rate * thin_cost +
		(control[7] * (state[0] + state[3]) + control[8] * (state[6] + state[9])) * protect_rate * protect_cost;

	if (dim_path_constraints > 1){
		path_constraints[1] = control[0] - parametrizedcontrol(
			1, n_stages, 0, &optimvars[0*n_stages], normalized_time,
			fixed_initial_time, fixed_final_time);

		path_constraints[2] = control[1] - parametrizedcontrol(
			1, n_stages, 0, &optimvars[1*n_stages], normalized_time,
			fixed_initial_time, fixed_final_time);

		path_constraints[3] = control[2] - parametrizedcontrol(
			1, n_stages, 0, &optimvars[2*n_stages], normalized_time,
			fixed_initial_time, fixed_final_time);

		path_constraints[4] = control[3] - parametrizedcontrol(
			1, n_stages, 0, &optimvars[3*n_stages], normalized_time,
			fixed_initial_time, fixed_final_time);

		path_constraints[5] = control[4] - parametrizedcontrol(
			1, n_stages, 0, &optimvars[4*n_stages], normalized_time,
			fixed_initial_time, fixed_final_time);

		path_constraints[6] = control[5] - parametrizedcontrol(
			1, n_stages, 0, &optimvars[5*n_stages], normalized_time,
			fixed_initial_time, fixed_final_time);

		path_constraints[7] = control[6] - parametrizedcontrol(
			1, n_stages, 0, &optimvars[6*n_stages], normalized_time,
			fixed_initial_time, fixed_final_time);

		path_constraints[8] = control[7] - parametrizedcontrol(
			1, n_stages, 0, &optimvars[7*n_stages], normalized_time,
			fixed_initial_time, fixed_final_time);

		path_constraints[9] = control[8] - parametrizedcontrol(
			1, n_stages, 0, &optimvars[8*n_stages], normalized_time,
			fixed_initial_time, fixed_final_time);

	}
}


