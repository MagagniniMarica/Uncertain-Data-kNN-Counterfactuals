# Source code and full experiment results for "Counterfactual explanations with the k-Nearest Neighborhood classifier and uncertain data"

This repository contains the source code used for the experiments in the paper:

> **"Counterfactual explanations with the k-Nearest Neighborhood classifier and uncertain data"**  
> by *Emilio Carrizosa, Renato De Leone, and Marica Magagnini*

---

The code in this repository implements the nonlinear optimization problem (15), as well as the heuristic algorithm proposed in the paper.  
It includes all datasets and supporting functions required to reproduce the experiments described in *Counterfactual explanations with the k-Nearest Neighborhood classifier and uncertain data*.

---

## Repository structure 

The repository is organized as follows:

- `Experiment_1/`: contains the Python scripts used to perform Experiment 1, described in Section 5.1 of the paper.
- `Experiment_2/`: contains the Python scripts used to perform Experiment 2, described in Section 5.2 of the paper. The subfolders contain the corresponding dataset information.
- `Experiment_3/`: contains the Python scripts used to perform Experiment 3, described in Section 5.3 of the paper. The subfolders contain the corresponding dataset information.
- `fun/`: contains the support code used by the experimental scripts. In particular, this folder includes the dataset-related scripts, the data-type classes, and the functions used to build and run the proposed method.

## Using the proposed method

Researchers who want to apply the proposed method to their own instances should use the code in the `fun/` folder. This folder contains the reusable implementation of the method, including the routines for defining the input data structures and constructing the optimization models. In particular, the main components are:

- `Dataset_selection`: selects the dataset to be used in the experiments or in a new test instance.
- `SymbolicDataset_`: transforms the selected dataset into the appropriate symbolic dataset representation, either using balls or rectangles.
- `Symbolic_US`: transforms both the input instance and the counterfactual instance into their symbolic representation.
- `kNN_rule`: implements the k-nearest-neighbour classification rule described in the paper.
- `end_counterf_`: builds the endogenous counterfactual.
- `Heuristic_counterf_`: instantiates the counterfactual optimization problem and solves it using the proposed heuristic method.
- `exact_COP_`: instantiates the counterfactual optimization problem and solves it exactly.

Therefore, the experimental scripts can also be used as examples of complete pipelines showing how to combine the functions in `fun/` to apply the proposed method to a dataset.

### Typical workflow
The main functions can be imported directly from the `fun` module: 

```python
from fun import Dataset_selection, SymbolicDataset_, Symbolic_US
from fun import kNN_rule, end_counterf_, exact_COP_, Heuristic_counterf_

# 1. Select or load the dataset using Dataset_selection.
# 2. Convert the dataset into a symbolic representation using SymbolicDataset_.
# 3. Use Symbolic_US to obtain the symbolic representation of the input instance
#    and to initialize the corresponding counterfactual representation.
# 4. Apply the kNN classification rule using kNN_rule to verify the classification
#    of the input instance.
# 5. Build the endogenous counterfactual using end_counterf_.
# 6. Solve the counterfactual optimization problem:
#    - with Heuristic_counterf_ for the heuristic method;
#    - with exact_COP_ for the exact method.
```
  
## Requirements and Python version

The code was tested with Python 3.12.3 and the package versions listed in REQUIREMENTS.txt. 

A valid Gurobi license is required to run the optimization models.
