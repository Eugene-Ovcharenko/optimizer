import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Union, Tuple, Optional, List
from pymoo.problems import get_problem
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize
from pymoo.termination.default import DefaultMultiObjectiveTermination
from pymoo.visualization.scatter import Scatter
from pymoo.decomposition.asf import ASF
from pymoo.indicators.hv import Hypervolume
from pymoo.core.result import Result
from test_problem import optimization_problem_test


# Problem implementation
class Problem(ElementwiseProblem):
    """
    A custom problem implementation for optimization, inheriting from pymoo's ElementwiseProblem.

    The class is initialized with parameters, objectives, and constraints
    to create a multi-objective optimization problem.
    The `_evaluate` method is used to compute objective and
    constraint values given a solution in the design space.

    Attributes:
        param_names (list): A list of parameter names.
        obj_names (list): A list of objective names.
        constr_names (list): A list of constraint names.
    """

    def __init__(self, parameters, objectives, constraints):
        """
        Initializes the Problem with given parameter bounds, objectives, and constraints.

        Args:
            parameters (dict): A dictionary where keys are parameter names and
                               values are tuples with lower and upper bounds.
            objectives (list): A list of objective names.
            constraints (list): A list of constraint names.
        """
        self.param_names = list(parameters.keys())
        lower_bounds = [parameters[name][0] for name in self.param_names]
        upper_bounds = [parameters[name][1] for name in self.param_names]
        super().__init__(n_var=len(parameters),  # calculated from the number of parameters
                         n_obj=len(objectives),  # calculated from the number of objectives
                         n_constr=len(constraints),  # calculated from the number of constraints
                         xl=np.array(lower_bounds),  # lower bounds from input
                         xu=np.array(upper_bounds)  # upper bounds from input
                         )
        self.obj_names = objectives
        self.constr_names = constraints

    def _evaluate(self, x, out, *args, **kwargs):
        """
        Evaluates the problem for a given solution, updating objective and constraint values.

        Args:
            x (np.ndarray): An array of parameter values representing a solution.
            out (dict): A dictionary to store output results. This will be updated with objective values ("F") and constraint values ("G").

        Returns:
            None: This function updates the `out` dictionary with the calculated objective and constraint values.
        """
        params = dict(zip(self.param_names, x))
        result = optimization_problem_test(params)
        out["F"] = np.array([result['objectives'][name] for name in self.obj_names])
        out["G"] = np.array([result['constraints'][name] for name in self.constr_names])


def extract_optimization_results(
        res: Result,
        problem: ElementwiseProblem,
        output_path: str
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    Optional[pd.DataFrame],
    Optional[pd.DataFrame],
    Optional[pd.DataFrame],
    Optional[pd.DataFrame]
]:
    """
    Extracts optimization results from a pymoo Result object into DataFrames,
    storing history to an Excel file and returning DataFrames for design space,
    objective spaces, constraints, CV, opt, and pop.

    Args:
        res (Result): The optimization result object from pymoo (pymoo.core.result.Result).
        problem (ElementwiseProblem): The problem instance with parameter names and constraints.
        output_path (str): The path to store the Excel file with history data.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame],
        Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
            - history_df: DataFrame containing the optimization history.
            - X: DataFrame containing the design space values.
            - F: DataFrame containing the objective space values.
            - G: DataFrame containing the constraint values (if available).
            - CV: DataFrame containing the aggregated constraint violation (if available).
            - opt: DataFrame containing the solutions as a Population object (if available).
            - pop: DataFrame containing the final Population (if available).
    """
    # Ensure the output path exists
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Create history DataFrame
    history_data = []
    for algo_index, algo in enumerate(res.history):
        for sol in algo.pop:
            record = {'generation': algo_index + 1}
            record.update(dict(zip(problem.param_names, sol.X)))
            record.update(dict(zip(problem.obj_names, sol.F)))
            if sol.has("G"):
                record.update(dict(zip(problem.constr_names, sol.G)))
            if sol.has("CV"):
                record['CV'] = sol.CV
            history_data.append(record)

    history_df = pd.DataFrame(history_data)

    # Save to Excel
    history_df.to_excel(os.path.join(output_path, 'history.xlsx'))

    # Create DataFrames for other components
    X = pd.DataFrame(res.X, columns=problem.param_names)
    F = pd.DataFrame(res.F, columns=problem.obj_names)

    G = pd.DataFrame() if not hasattr(res, 'G') or res.G is None else pd.DataFrame(res.G, columns=problem.constr_names)
    CV = pd.DataFrame() if not hasattr(res, 'CV') or res.CV is None else pd.DataFrame(res.CV, columns=['CV'])

    # Extract opt and pop DataFrames
    opt_df = None
    if res.opt is not None:
        opt_data = [{'X': dict(zip(problem.param_names, ind.X)),
                     'F': dict(zip(problem.obj_names, ind.F)),
                     'G': dict(zip(problem.constr_names, ind.G)) if ind.has("G") else None,
                     'CV': ind.CV if ind.has("CV") else None} for ind in res.opt]
        opt_df = pd.DataFrame(opt_data)

    pop_df = None
    if res.pop is not None:
        pop_data = [{'X': dict(zip(problem.param_names, ind.X)),
                     'F': dict(zip(problem.obj_names, ind.F)),
                     'G': dict(zip(problem.constr_names, ind.G)) if ind.has("G") else None,
                     'CV': ind.CV if ind.has("CV") else None} for ind in res.pop]
        pop_df = pd.DataFrame(pop_data)

    return (history_df, X, F, G, CV, opt_df, pop_df)


def create_pareto_front_plot(
        data: pd.DataFrame,
        folder_path: str,
        pymoo_problem: str = "welded_beam"
) -> None:
    """
    Creates and saves a scatter plot showing the Pareto front for a specified problem along with additional data.

    Args:
        data (pd.DataFrame): A DtaFrame with objective values.
        folder_path (str): The folder path where the plot will be saved.
        pymoo_problem (str): The name of the problem to get the Pareto front from. Defaults to "welded_beam".

    Returns:
        None: The function saves the plot to the specified folder and does not return any value.
    """
    pymoo_scatter_plot = Scatter(title="Pareto front for welded beam problem")
    pymoo_scatter_plot.add(get_problem(pymoo_problem).pareto_front(use_cache=False), plot_type="line", color="black")
    pymoo_scatter_plot.add(data.values, facecolor="none", edgecolor="red", alpha=0.8, s=20)
    pymoo_scatter_plot.save(os.path.join(folder_path, 'pareto_front.png'))


def plot_objective_minimization(
        history_df: pd.DataFrame,
        folder_path: str
) -> None:
    """
    Creates a plot showing the minimum objective values over generations and saves it to a specified folder.

    Args:
        history_df (pd.DataFrame): The DataFrame containing optimization history.
        folder_path (str): The directory path to save the plot.

    Returns:
        None: This function creates a Seaborn plot and saves it to the specified file.
    """
    objective_columns = history_df.filter(like='objective').columns
    objectives_min_per_generation = history_df.groupby('generation')[objective_columns].min()
    objectives_over_time = objectives_min_per_generation.min(axis=1).tolist()
    sns.set(style="whitegrid")
    plt.figure(figsize=(7, 5))
    sns.lineplot(
        x=range(1, len(objectives_over_time) + 1),
        y=objectives_over_time,
        marker='.',
        linestyle='-',
        color='b',
        markersize=10
    )
    plt.title("Objective Minimization Over Generations")
    plt.xlabel("Generation")
    plt.ylabel("Objective Value")
    plt.tight_layout()
    plt.savefig(os.path.join(folder_path, 'convergence_by_objectives.png'))


def find_best_tradeoff(
        F: pd.DataFrame,
        folder_path: str,
        objectives_weights: List[float] = [0.5, 0.5]
) -> int:
    """
    Finds the best trade-off between two objectives using Augmented Scalarization Function (ASF) and
    creates a scatter plot, highlighting the optimal point.

    Args:
        F (pd.DataFrame): DataFrame containing the objective values.
        folder_path (str): The directory path to save the plot.
        objectives_weights (List[float]): A list of weights for the objectives to be used in ASF. Defaults [0.5, 0.5].

    Returns:
        int: The index of the optimal point with the best trade-off (for objective values DataFrame F).
    """
    if not np.isclose(sum(objectives_weights), 1):
        raise ValueError("The sum of objectives_weights must be 1.")
    approx_ideal = F.min(axis=0)
    approx_nadir = F.max(axis=0)
    nF = (F - approx_ideal) / (approx_nadir - approx_ideal)
    nFm = nF.to_numpy(copy=True)
    weights = np.array(objectives_weights)
    decomp = ASF()
    best_index = decomp.do(nFm, 1 / weights).argmin()

    plt.figure(figsize=(7, 5))
    sns.scatterplot(data=F, x=F.columns[0], y=F.columns[1],  label="All points",
                    s=30, color='blue', edgecolor='blue', alpha=0.6)
    sns.scatterplot(data=F.iloc[[best_index]], x=F.columns[0], y=F.columns[1],  label="Best point",
                    s=200, marker="x", color="red")
    plt.title("Objective Space")
    plt.xlabel(F.columns[0])
    plt.ylabel(F.columns[1])
    plt.tight_layout()
    plt.savefig(os.path.join(folder_path, 'objective_space.png'))

    return best_index


def plot_convergence_by_hypervolume(
        history_df: pd.DataFrame,
        folder_path: str,
        ref_point: np.ndarray = np.array([1.1, 1.1]),
) -> None:
    """
    Plots convergence by hypervolume and saves it to a specified folder.

    Args:
        history_df (pd.DataFrame): The DataFrame containing optimization history.
        folder_path (str): The directory path to save the plot.
        ref_point (np.ndarray, optional): The reference point for
                   hypervolume calculation. Defaults to np.array([1.1, 1.1]).

    Returns:
        None: This function creates a plot and saves it to the specified file.
    """
    approx_ideal = history_df.filter(like='objective').min(axis=0)
    approx_nadir = history_df.filter(like='objective').max(axis=0)
    hv_metric = Hypervolume(
        ref_point=ref_point,
        ideal=approx_ideal,
        nadir=approx_nadir,
        zero_to_one=True,
        norm_ref_point=False
    )
    hist_F = [history_df[history_df['generation'] == gen].filter(like='objective').to_numpy()
              for gen in history_df['generation'].unique()]
    hypervolume_values = [hv_metric.do(_F) for _F in hist_F]
    n_evals = list(range(1, len(hypervolume_values) + 1))

    sns.set(style="whitegrid")
    plt.figure(figsize=(7, 5))
    sns.lineplot(x=n_evals, y=hypervolume_values, marker='.', linestyle='-', color='b', markersize=10)
    plt.title("Convergence by Hypervolume")
    plt.xlabel("Function Evaluations")
    plt.ylabel("Hypervolume")
    plt.tight_layout()
    plt.savefig(os.path.join(folder_path, 'convergence_by_hypervolume.png'))


if __name__ == "__main__":

    folder_path = 'results'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # parameter boundaries (min, max)
    parameters = {
        'param1': (0.01, 10.0),
        'param2': (0.01, 10.0),
        'param3': (0.01, 10.0),
        'param4': (0.01, 10.0)
    }

    # objectives names
    objectives = ['objective1', 'objective2']

    # constraints names
    constraints = ['constraint1', 'constraint2', 'constraint3', 'constraint4']

    # problem initialization
    problem = Problem(parameters, objectives, constraints)

    # algorithm initialization
    algorithm = NSGA2(
        pop_size=40,
        n_offsprings=10,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=0.7, eta=30),
        mutation=PM(prob=0.2, eta=25),
        eliminate_duplicates=True
    )

    # termination criteria
    termination = DefaultMultiObjectiveTermination(
        xtol=1e-8,
        cvtol=1e-6,
        ftol=0.0025,
        period=30,
        n_max_gen=100,
        n_max_evals=100000
    )

    # run optimization
    res = minimize(problem,
                   algorithm,
                   termination,
                   seed=1,
                   save_history=True,
                   verbose=True)

    # result storage
    history_df, X, F, G, CV, opt, pop = extract_optimization_results(res, problem, folder_path)

    # Pareto front for "welded beam" problem
    create_pareto_front_plot(F, folder_path, pymoo_problem="welded_beam")

    # Objective Minimization Over Generations evaluation
    plot_objective_minimization(history_df, folder_path)

    # Find the best trade-off between two objectives F1 and F2 using Augmented Scalarization Function (ASF)
    i = find_best_tradeoff(F, folder_path, objectives_weights=[0.5, 0.5])
    print(f'Best regarding ASF:\nPoint #{i}\n{F.iloc[i]}')

    # Convergence by Hypervolume
    plot_convergence_by_hypervolume(history_df, folder_path, ref_point=np.array([1.1, 1.1]))





    # TODO: design space
    # TODO: constrains space
    # TODO: all objectives by time oe epoch
    # TODO: save table of verbose
    # TODO: network diagram
