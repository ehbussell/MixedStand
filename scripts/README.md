# Mixed Stand Analysis Scripts

These python scripts run the main analyses investigating optimal control of sudden oak death in mixed species forest stands. Each file can be run from the command line. For details of command line arguments for each script, run: ```python path/to/script -h``` from the root project directory. Data and figures will be generated in new folders, note that some analyses are computationally intensive.

The scripts perform the following analyses:
1. [budget_scan.py](budget_scan.py):        Scan over maximum budget for control, comparing optimal strategies and performance
1. [div_cost_scan.py](div_cost_scan.py):    Scan over the relative benefit of conserving biodiversity
1. [global_optimal](global_optimal):        Re-scale the approximate model to match the optimal open-loop strategy, and test optimal control performance with the newly scaled model
1. [obs_uncertainty.py](obs_uncertainty.py):    Analyse the effect of imperfect observation at MPC update steps
1. [ol_mpc_control.py](ol_mpc_control.py):  Optimise control using the open-loop and MPC frameworks, comparing different MPC update periods
1. [param_sensitivity.py](param_sensitivity.py):    Test how dynamics and optimal control strategies depend on parameterisation
1. [param_uncertainty.py](param_uncertainty.py):    Test how open-loop and MPC frameworks perform when there is parameter uncertainty
1. [scale_and_fit.py](scale_and_fit.py):    Scale simulations to match Cobb *et al.* (2012) and fit approximate model ready for control optimisation