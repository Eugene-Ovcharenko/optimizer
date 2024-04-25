# Multi-Objective Optimization with Pymoo / PHV_optimizer /

This repository contains a multi-objective optimization script for prosthetic heart valve leaflet design improvement
using the [pymoo](https://pymoo.org) library. The custom problem is described by parameters, multiple objectives, and
constraints. The algorithm used is **NSGA-II**.

## Table of Contents

* [Key Components](#key-components)
* [Problem Definition](#problem-definition)
* [Optimization Process](#optimization-process)
* [Visualization Functions](#visualization-functions)
* [Dependencies](#dependencies)
* [<font color="yellow">TODO</font>](#TODO)

## Key Components

- **optimizer.py**: Contains an optimization algorithm based on the class `Problem`. Custom class `Problem`  (derived
  from pymoo's `ElementwiseProblem`) defines objectives, parameters, and constraints, along with the `_evaluate` method
  to calculate them.
- **test_problem.py**: Contains `optimization_problem_test`, which evaluates a specific problem ("welded_beam")
  based on given parameters and returns objective and constraint results.
- **visualization.py**: Contains functions to create various plots, such as Pareto fronts, objective convergence,
  hypervolume, scatter plots, and parallel coordinates.

## Problem Definition

The custom problem class `Problem` is defined with:

- **Parameters**: Dictionary of parameter names and their lower and upper bounds (e.g., `'param1': (0.01, 10.0)`).
- **Objectives**: List of objective names to optimize (e.g., `['objective1', 'objective2']`).
- **Constraints**: List of constraint names representing conditions to meet
  (e.g., `['constraint1', 'constraint2', 'constraint3', 'constraint4']`):
    - Constraints are defined numerically, where each constraint function outputs a value representing the degree of
      violation.
    - A constraint value less than or equal to zero (<= 0) indicates that the constraint is satisfied (non-violated).
    - A positive constraint value (> 0) indicates a violation.

## Optimization Process

The optimization is using the following components:

- **Algorithm Initialization**: Defines population size, crossover, mutation, and other parameters.
- **Termination Criteria**: Conditions for ending the optimization, such as maximum generations and evaluations.
- **Results Extraction**: A function `extract_optimization_results` that extracts optimization results into DataFrames
  for analysis and storage.

## Visualization Functions

The repository contains several visualization functions to analyze the optimization results:

- **Pareto Front**: Shows the Pareto-optimal solutions.
- **Objective Convergence**: Displays how objectives improve over generations.
- **Hypervolume**: Shows convergence by hypervolume.
- **Scatter Plots**: Plots objectives against parameters and constraints.
- **Parallel Coordinates**: Plots multiple variables on parallel axes to understand relationships.

## Dependencies

Necessary dependencies are listed in `requirements.txt`.

---

## <font color="yellow">TODO</font>

- <font color="yellow">Save the table of verbose output.</font>
- <font color="yellow">Organize logging effectively.</font>
- <font color="yellow">Generate all plots after calculations from saved data.</font>
- <font color="yellow">Try R-NSGA-II.</font>
- <font color="yellow">Try NSGA-III.</font>
- <font color="yellow">Try U-NSGA-III.</font>
- <font color="yellow">Try R-NSGA-III.</font>
- <font color="yellow">Try MOEA/D.</font>

---

For more information about pymoo and its multi-objective optimization features, refer to the [pymoo documentation]
(https://pymoo.org/documentation.html).

