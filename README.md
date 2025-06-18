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


## Configuration file

Configure optimization with `.yaml` files located in `./configuration` folder.

List of parameters:
 - `parameters:` - **mandatory**
   - `param 1: [min, max]`
   - `param 2: [min, max]`
   - .....
   - `param N: [min, max]`
 - `objectives:` - **mandatory**
   - `- objective 1`
   - `- objective 2`
   - .....
   - `- objective N`
 - `constrains:` - optional
   - `- constr 1`
   - `- constr 2`
   - .....
   - `- constr N`
 - `problem_definition:`
   - `name: XXXX` - *XXXX* - name of your problem. Optional.
   - `position: XXXX` - **mandatory!** *XXXX* may be in `[ao, mitr]`. This affect to boundary conditions
   - `problem_name: XXXX` - **mandatory!** *XXXX* may be `[leaflet_contact, leaflet_single, test]`
   - `DIA: XX` - *XX* is diameter of lealet apparatus. Measured in mm.
   - `Lift: XX` - how far leaflet would be lifted to simulate frame. Fully optional, by default assumed by 0 mm
   - `SEC: XX` - sector of circle occupied by one leaflet
   - `mesh_step: XX` - size of the mesh. Used in leaflet points generation. Default value - 0.35. More value - coarser mesh
   - `material:` - **mandatory!** Following part of yaml defining material properties
     - `material_definition_type: XX` - **mandatory!** Define type of used material model: `[linear, polynomial, ortho`
     - `material_name: XX` - just name of used material, small QoL
     - `poisson_coeff: XX` - Poisson coefficient. Used with `linear` or `polynomial` model
     - `Dens: XX` - **mandatory!** Density of material. By default - `1e-9 tonn/mm`
     - `s_lim: XX` - UTS for material
     - `material_csv_path: XX` - name of the csv-file located in `./configuration` folder. Used with `polynomial` material model. Format - `stress, strain`
     - `ortho_coeffs_E:` - this is array of Young's modulus for `ortho` material model. Using cylindrical coodrinate system in this point 
       - ` - E1` - Young's modulus in `radial` direction
       - ` - E1` - Young's modulus in `circumferential` direction
       - ` - E1` - Young's modulus in `Z` direction
     - `ortho_coeffs_poisson:` - this is array of Poisson coefficients for `ortho` material model. Using cylindrical coodrinate system in this point 
       - ` - p1` - Poisson coefficients in `radial` direction
       - ` - p2` - Poisson coefficients in `circumferential` direction
       - ` - p3` - Poisson coefficients in `Z` direction
   - `Abaqus:` - FEA related part of configuration file
     - `abq_cpus: XX` - how much cpus were used for FEA
     - `tangent_behavior: XX` - tangential stiffness used in contact problem `leaflet_contact`
     - `normal_behavior: XX` - normal stiffness used in contact problem `leaflet_contact`
   - `optimizer:` - optimizer-related parameters. everything here is **mandatory!**. Read PyMoo manuals
     - `pop_size: XX`
     - `offsprings: XX`
     - `crossover_chance: XX`
     - `mutation_chance: XX`
     - `crossover_eta: XX`
     - `mutation_eta: XX`
     - `termination_parameters:`
       - `xtol: XX`
       - `cvtol: XX`
       - `ftol: XX`
       - `period: XX`
       - `n_max_gen: XX`
       - `n_max_evals: XX`
 
    You can add additional **Hydra**-related parameters below.
---

## <font color="yellow">TODO</font>
- <font color="yellow">Try R-NSGA-II.</font>
- <font color="yellow">Try NSGA-III.</font>
- <font color="yellow">Try U-NSGA-III.</font>
- <font color="yellow">Try R-NSGA-III.</font>
- <font color="yellow">Try MOEA/D.</font>

---

For more information about pymoo and its multi-objective optimization features, refer to the [pymoo documentation]
(https://pymoo.org/documentation.html).

