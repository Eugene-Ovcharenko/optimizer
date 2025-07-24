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

---
# Mathematical Basis of the Unfolding Algorithm <!-- README.md -->

## 1. Developability and Gaussian Curvature

For a smooth surface $S \subset \mathbb{R}^3$ the Gaussian curvature is the product of the principal curvatures:

$$
K(p)=\kappa_1(p)\,\kappa_2(p).
$$

* **Gauss’ Theorema Egregium.** Gaussian curvature is invariant under isometric deformations. Hence a surface is *developable* (can be unfolded to the plane without stretching) iff $K\equiv0$.

## 2. “Almost‑Developable” Engineering Approximation

In practice we admit small curvatures $|K|\le |K|_{\max}$.
Let $a$ be the radius of a small circular patch. For sufficiently small $a$

$$
\frac{\delta A}{A}\;\approx\;\frac{K\,a^{2}}{6},
\tag{1}
$$

where $\delta A/A$ is the relative area change when the patch is flattened.

### Practical design formula

Using (1) and setting the characteristic linear size $L=2a$,

$$
|K|_{\max}\;=\;\frac{24\,(\delta A/A)_{\max}}{L^{2}}.
\tag{2}
$$

Equation(2) underlies `gaussian_tolerance_from_area_strain`, converting an admissible area strain percentage into a tolerance for $|K|$.

## 3. Engineering Interpretation of Tolerances

delta(A)/A_max | abs(K_max) at L=12 mm, mm^-2 | Equivalent radius R=sqrt(1/abs(K)) | Typical linear strain
--- | --- | --- | ---
1%(0.01) | $$1.4\times10^{-3}$$ | 31.6mm | ≈0.5%
3%(0.03) | $$4.2\times10^{-3}$$ | 17.3mm | ≈1.5%
6%(0.06) | $$8.3\times10^{-3}$$ | 11.0mm | ≈3%
10%(0.10) | $$1.4\times10^{-2}$ $| 8.4mm | ≈5%

\*Linear strain is approximated by $\tfrac12\,\delta A/A$ for small deformations.

* **Polymeric leaflets** (Formlabs Elastic50A, ShoreA50) sustain ≥100% elastic elongation\[3]; thus $|K|\le10^{-2}\,\text{mm}^{-2}$ (≤6% area strain) is mechanically safe and geometrically accurate.
* A tolerance of $10^{-1}\,\text{mm}^{-2}$ corresponds to 60% area change and is acceptable only when large material draw‑in is permissible.

## 4. Tolerance Selection Algorithm in Code

```python
# Pseudocode (see adaptive_tolerance in /gaussian_curvature_v2.py)

tol = gaussian_tolerance_from_area_strain(
        diameter_mm=L,
        max_area_strain=desired_area_strain)

result = evaluate_developability(mesh, tol)
```

`adaptive_tolerance` iterates over a set of allowable $\delta A/A$ values (1%,3%,6%,10%) and returns the smallest $|K|_{\max}$ that passes `evaluate_developability`. This yields the **least‑distorting** geometry consistent with manufacturability.


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

