import os
import sys
import time
import datetime
import logging
from typing import Tuple, Optional, List, Dict
import numpy as np
import pandas as pd
import openpyxl
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.result import Result
from pymoo.decomposition.asf import ASF
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize
from pymoo.termination.default import DefaultMultiObjectiveTermination

from test_problem import optimization_problem_test
from visualization import *


class MultiStreamHandler(object):
    """
    A custom stream handler that writes to multiple outputs (console and file).
    """

    def __init__(self, *streams):
        self.streams = streams

    def write(self, buf):
        # Write the output to all streams
        for stream in self.streams:
            stream.write(buf)
            stream.flush()  # Ensure output is immediately written to the stream

    def flush(self):
        for stream in self.streams:
            if stream and not stream.closed:
                try:
                    stream.flush()
                except Exception as e:
                    logger.error(f"Error while flushing stream: {e}")


def setup_logger(
        folder_path: str,
        log_file_name: str = 'terminal_log.txt'
) -> logging.Logger:
    """
    Sets up a logger to capture verbose output, writing to both a log file and the console.

    Args:
        folder_path (str): The directory path to save the log file.
        log_file_name (str, optional): The name of the log file. Defaults to 'terminal_log.txt'.

    Returns:
        logging.Logger: Configured logger for capturing terminal output.
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    logger = logging.getLogger('multi_stream_logger')
    logger.setLevel(logging.INFO)

    # Create a file handler for logging to a file
    file_handler = logging.FileHandler(os.path.join(folder_path, log_file_name))
    file_handler.setLevel(logging.INFO)

    # Create a console handler for logging to the console
    console_handler = logging.StreamHandler()  # Outputs to the console
    console_handler.setLevel(logging.INFO)

    # Define the log output format
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # Redirect stdout and stderr to multi-stream handler
    multi_stream = MultiStreamHandler(sys.stdout, file_handler.stream)
    sys.stdout = multi_stream  # Redirect standard output
    sys.stderr = multi_stream  # Redirect standard error

    return logger


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
            out (dict): A dictionary to store output results. This will be updated with objective values ("F")
            and constraint values ("G").

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
    history_df.to_excel(os.path.join(output_path, 'history.xlsx'))

    X = pd.DataFrame(res.X, columns=problem.param_names)

    F = pd.DataFrame(res.F, columns=problem.obj_names)

    G = pd.DataFrame() if not hasattr(res, 'G') or res.G is None else pd.DataFrame(res.G, columns=problem.constr_names)

    CV = pd.DataFrame() if not hasattr(res, 'CV') or res.CV is None else pd.DataFrame(res.CV, columns=['CV'])

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
    sns.scatterplot(data=F, x=F.columns[0], y=F.columns[1], label="All points",
                    s=30, color='blue', edgecolor='blue', alpha=0.6)
    sns.scatterplot(data=F.iloc[[best_index]], x=F.columns[0], y=F.columns[1], label="Best point",
                    s=200, marker="x", color="red")
    plt.title("Objective Space")
    plt.xlabel(F.columns[0])
    plt.ylabel(F.columns[1])
    plt.tight_layout()
    plt.savefig(os.path.join(folder_path, 'objective_space.png'))

    return best_index


def print_algorithm_params(algorithm_params: dict) -> dict:
    """
    Prints detailed algorithm parameters, including additional information
    for crossover and mutation, line by line, and returns the detailed parameters.

    Args:
        algorithm_params (dict): The algorithm parameters as a dictionary.

    Returns:
        dict: A dictionary containing the detailed parameters with additional information
              for crossover and mutation.
    """
    detailed_params = {
        **algorithm_params,
        "crossover": {
            "method": algorithm_params["crossover"].__class__.__name__,
            "prob": algorithm_params["crossover"].prob.value,
            "eta": algorithm_params["crossover"].eta.value
        },
        "mutation": {
            "method": algorithm_params["mutation"].__class__.__name__,
            "prob": algorithm_params["mutation"].prob.value,
            "eta": algorithm_params["mutation"].eta.value
        }
    }

    print("Algorithm Parameters:")
    for key, value in detailed_params.items():
        if isinstance(value, dict):
            print(f"{key}:")
            for sub_key, sub_value in value.items():
                print(f"  {sub_key}: {sub_value}")
        else:
            print(f"{key}: {value}")

    return detailed_params


def print_termination_params(termination_params: dict) -> None:
    """
    Prints detailed termination parameters, line by line.

    Args:
        termination_params (dict): The termination parameters as a dictionary.

    Returns:
        None: This function prints details line by line.
    """
    print("Termination Parameters:")
    for key, value in termination_params.items():
        print(f"{key}: {value}")


def create_results_folder(base_folder: str = 'results') -> str:
    """
    Creates a subfolder within a specified base folder to store results.
    The subfolder name is based on the current date and an incrementing number.

    Args:
        base_folder (str): The base folder where subfolders will be created. Defaults to 'results'.

    Returns:
        str: The path to the created subfolder.
    """
    # Ensure the base folder exists
    if not os.path.exists(base_folder):
        os.makedirs(base_folder)

    # Get the list of existing subfolders
    existing_subfolders = [f for f in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, f))]

    # Determine the next subfolder number and format the current date
    subfolder_number = len(existing_subfolders) + 1
    current_time = datetime.datetime.now()
    formatted_date = current_time.strftime('%d_%m_%Y')

    # Create the new subfolder name
    subfolder_name = f"{subfolder_number:03}_{formatted_date}"
    folder_path = os.path.join(base_folder, subfolder_name)

    # Create the subfolder if it doesn't exist
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Optionally print a confirmation message
    print(f"Created folder: {folder_path}")

    # Return the path to the created subfolder
    return folder_path


def find_best_result(F: pd.DataFrame, weights: list[float]) -> int:
    """
    Finds the best result in a multi-objective optimization based on ASF.

    Args:
        F (pd.DataFrame): DataFrame containing the objective values.
        weights (list[float]): A list of weights to assign to each objective.

    Returns:
        int: The index of the optimal solution in the DataFrame.
    """
    # Check if the sum of weights is approximately 1
    if not np.isclose(sum(weights), 1):
        raise ValueError("Weights must sum to 1.")

    # Normalize objective values to find ideal and nadir points
    approx_ideal = F.min(axis=0)
    approx_nadir = F.max(axis=0)

    # Normalized F with ideal and nadir points
    nF = (F - approx_ideal) / (approx_nadir - approx_ideal)
    nFm = nF.to_numpy(copy=True)
    decomp = ASF()
    best_index = decomp.do(nFm, 1 / np.array(weights)).argmin()

    return best_index


def save_optimization_summary(
        folder_path: str,
        best_index: int,
        elapsed_time: float,
        F: pd.DataFrame,
        X: pd.DataFrame,
        G: pd.DataFrame,
        history_df: pd.DataFrame,
        termination_params: dict,
        detailed_algo_params: dict
) -> None:
    """
    Save optimization summary to an Excel file with given details, including termination and detailed algorithm parameters.

    Args:
        folder_path (str): The path to the folder where results are stored.
        best_index (int): The index of the best optimization result.
        elapsed_time (float): Time taken for the optimization.
        F (pd.DataFrame): DataFrame with objective values.
        X (pd.DataFrame): DataFrame with parameter values.
        G (pd.DataFrame): DataFrame with constraint values.
        history_df (pd.DataFrame): DataFrame containing the optimization history.
        termination_params (dict): Dictionary of termination parameters.
        detailed_algo_params (dict): Dictionary of detailed algorithm parameters.

    Returns:
        None: The function saves the summary to an Excel file in the 'results' folder.
    """
    base_folder = folder_path.split(os.path.sep)[0]
    summary_file = os.path.join(base_folder, 'optimization_summary.xlsx')

    if os.path.exists(summary_file):
        workbook = openpyxl.load_workbook(summary_file)
        worksheet = workbook.active
    else:
        workbook = openpyxl.Workbook()
        worksheet = workbook.active

        headers_algo_params = []
        for key, value in detailed_algo_params.items():
            if isinstance(value, dict):
                headers_algo_params.extend([f"{key}_{sub_key}" for sub_key in value.keys()])
            else:
                headers_algo_params.append(f"{key}")
        headers = (["Timestamp", "Generation", "Best Index", "Elapsed Time"] + \
                   [f"F_{col}" for col in F.columns] + \
                   [f"X_{col}" for col in X.columns] + \
                   [f"G_{col}" for col in G.columns] + \
                   [f"Term_{key}" for key in termination_params] + \
                   headers_algo_params + \
                   ['Folder_path'])
        worksheet.append(headers)

    generation_number = history_df['generation'].max()

    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    best_F = F.iloc[best_index].tolist()
    best_X = X.iloc[best_index].tolist()
    best_G = G.iloc[best_index].tolist()

    termination_values = [value for key, value in termination_params.items()]

    detailed_algo_values = []  # Add detailed algorithm parameters to the new row
    for key, value in detailed_algo_params.items():
        if isinstance(value, dict):
            # If it's a dictionary, iterate through its items and convert to string
            detailed_algo_values.extend([str(sub_value) for sub_key, sub_value in value.items()])
        elif isinstance(value, FloatRandomSampling):
            # Check if the object is a FloatRandomSampling and use its 'name' attribute
            detailed_algo_values.append(value.name)
        else:
            # Otherwise, convert the value to a string
            detailed_algo_values.append(str(value))

    new_row = [timestamp, generation_number, best_index, elapsed_time] + \
              best_F + best_X + best_G + termination_values + detailed_algo_values + [folder_path]

    worksheet.append(new_row)
    workbook.save(summary_file)


if __name__ == "__main__":
    # folder to store results
    folder_path = create_results_folder(base_folder='results')

    # logging
    logger = setup_logger(folder_path)
    start_time = time.time()
    logger.info("Starting optimization...")

    # parameter boundaries (min, max)
    parameters = {
        'param1': (0.01, 10.0),
        'param2': (0.01, 10.0),
        'param3': (0.01, 10.0),
        'param4': (0.01, 10.0)
    }
    print('Parameters:', parameters)

    # objectives names
    objectives = ['objective1', 'objective2']
    print('Objectives:', objectives)

    # constraints names
    constraints = ['constraint1', 'constraint2', 'constraint3', 'constraint4']
    print('Constraints:', constraints)

    # problem initialization
    problem = Problem(parameters, objectives, constraints)

    # algorithm initialization
    algorithm_params = {
        "pop_size": 40,
        "n_offsprings": 10,
        "sampling": FloatRandomSampling(),
        "crossover": SBX(prob=0.7, eta=30),
        "mutation": PM(prob=0.2, eta=25),
        "eliminate_duplicates": True
    }
    algorithm = NSGA2(**algorithm_params)
    detailed_algo_params = print_algorithm_params(algorithm_params)

    # termination criteria
    termination_params = {
        "xtol": 1e-8,
        "cvtol": 1e-6,
        "ftol": 0.0025,
        "period": 30,
        "n_max_gen": 10,
        "n_max_evals": 100000
    }
    termination = DefaultMultiObjectiveTermination(**termination_params)
    print_termination_params(termination_params)

    # run optimization
    res = minimize(problem,
                   algorithm,
                   termination,
                   seed=1,
                   save_history=True,
                   verbose=True)
    elapsed_time = time.time() - start_time

    # result storage
    history_df, X, F, G, CV, opt, pop = extract_optimization_results(res, problem, folder_path)
    history_df.to_csv(os.path.join(folder_path, 'history.csv'))
    X.to_csv(os.path.join(folder_path, 'X.csv'))
    F.to_csv(os.path.join(folder_path, 'F.csv'))
    G.to_csv(os.path.join(folder_path, 'G.csv'))
    CV.to_csv(os.path.join(folder_path, 'CV.csv'))
    opt.to_csv(os.path.join(folder_path, 'opt.csv'))
    pop.to_csv(os.path.join(folder_path, 'pop.csv'))

    #  Find the best trade-off between objectives using Augmented Scalarization Function (ASF)
    weights = [0.5, 0.5]
    best_index = find_best_result(F, weights)
    print(f'Best regarding ASF:\nPoint #{best_index}\n{F.iloc[best_index]}')
    print('Elapsed time:', elapsed_time)
    logger.info("Optimization completed.")

    # upload result to integrate table
    save_optimization_summary(
        folder_path,
        best_index,
        elapsed_time,
        F,
        X,
        G,
        history_df,
        termination_params,
        detailed_algo_params
    )

    # # Find the best trade-off between two objectives F1 and F2 using Augmented Scalarization Function (ASF)
    # i = find_best_tradeoff(F, folder_path, objectives_weights=[0.5, 0.5])
    # print(f'Best regarding ASF:\nPoint #{i}\n{F.iloc[i]}')
    # logger.info("Optimization completed.")
    #
    # # Pareto front for "welded beam" problem
    # create_pareto_front_plot(F, folder_path, pymoo_problem="welded_beam")
    #
    # # Convergence by Hypervolume
    # plot_convergence_by_hypervolume(history_df, folder_path, ref_point=np.array([100.0, 0.1]))
    #
    # # Convergence for objectives
    # plot_objective_convergence(history_df, folder_path)
    #
    # # Objectives vs parameters
    # plot_objectives_vs_parameters(X, F, folder_path)
    #
    # # Constrains vs parameters
    # plot_constrains_vs_parameters(X, G, folder_path)
    #
    # # Parallel coordinates plot
    # plot_parallel_coordinates(X, G, F, folder_path)
