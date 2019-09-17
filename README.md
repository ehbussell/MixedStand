# Mixed Stand Model

This project implements an epidemic model of forest dynamics in a mixed species forest stand under threat from sudden oak death (SOD). The model is based on work by CObb *et al.* (2012). The model is used to optimise management of SOD in order to protect the valuable tanoak species whilst conserving biodiversity.

## Prerequisites

### BOCOP
We use the software package BOCOP v2.0.5 for optimisation of control on the approximate models. This must be installed on your machine to use this functionality. Installation instructions can be found on the BOCOP website (http://www.bocop.org/).

The optimisation code for the two approximate models must be built first to provide a BOCOP executable.
```
cd path/to/mixed_stand_model/BOCOP
mkdir build
cmake -G "MSYS Makefiles" -DPROBLEM_DIR:PATH=path/to/mixed_stand_model/BOCOP path/to/BOCOP_Installation -DCMAKE_BUILD_TYPE=Release
```

### Python
We use python v3.7.3 available from https://www.python.org/.

## Folder Descriptions

There are 3 folders in this project:
1. mixed_stand_model:   Contains model and optimisation implementation
1. tests:               Contains tests to verify dynamics and optimiser are working correctly. Can be run from the project root directory using [pytest](https://docs.pytest.org/en/latest/) with the command ```python -m pytest .\tests\ -v```
1. scripts:             Contains python scripts that run the main analyses. See [README.md](scripts/README.md) in this folder for further details
