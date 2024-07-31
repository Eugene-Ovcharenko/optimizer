import gc
import os
import sys
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
from utils.global_variable import get_problem_name, get_percent, get_cpus
from utils.visualize import *
from utils.problem import init_procedure, Procedure
import time
from random import random
from glob2 import glob
from utils.global_variable import (set_problem_name, set_percent, set_cpus, set_base_name, set_s_lim, get_s_lim, set_id,
                                   set_dead_objects, set_mesh_step)
import pickle


class MultiStreamHandler:
    """
    A custom stream handler that writes to multiple outputs (console and file),
    while filtering out messages marked as red.
    """

    def __init__(self, *streams):
        self.streams = streams

    def write(self, buf):
        if not buf.strip().startswith('[RED]'):
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
                    logging.error(f"Error while flushing stream: {e}")


def setup_logger(folder_path: str, log_file_name: str = 'terminal_log.txt') -> logging.Logger:
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

    file_handler = logging.FileHandler(os.path.join(folder_path, log_file_name))
    file_handler.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()  # Outputs to the console
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    multi_stream = MultiStreamHandler(sys.stdout, file_handler.stream)
    sys.stdout = multi_stream  # Redirect standard output
    sys.stderr = multi_stream  # Redirect standard error

    return logger


def cleanup_logger(logger):
    """
    Cleans up the logger by removing handlers and restoring original stdout and stderr.
    """
    for handler in logger.handlers:
        handler.close()
        logger.removeHandler(handler)
    logger.handlers = []


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
    problem = None

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
        # result = Procedure.run_procedure(self=problem, params=x)
        self.problem = init_procedure(np.array(x))
        problem_name = get_problem_name().lower()
        cpus = get_cpus()
        parameters = np.array(x)
        if problem_name == 'beam':
            result = Procedure.run_procedure(self=self.problem, params=parameters)
            objective_values = result.get('objectives')
            objectives_dict = {
                "Displacement": objective_values.get('Displacement'),
                "Mass": objective_values.get('Mass')
            }
            constraint_values = result.get('constraints')
            constraints_dict = {
                "THK_constr": constraint_values.get('THK_constr'),
                "Width_constr": constraint_values.get('Width_constr'),
                "Smax_constr": constraint_values.get('Smax_constr')
            }
        elif problem_name == 'leaflet_single':
            result = Procedure.run_procedure(self=self.problem, params=parameters)
            objective_values = result['objectives']
            objectives_dict = {
                "1 - LMN_open": objective_values['1 - LMN_open'],
                "LMN_open": objective_values['LMN_open'],
                "LMN_closed": objective_values['LMN_closed'],
                "Smax": objective_values['Smax']
            }
            # print(f'obj: {objectives_dict}')
            constraint_values = result['constraints']
            constraints_dict = {
                "VMS-Smax": constraint_values['VMS-Smax']
            }
            # print(f'cons: {constraints_dict}')
        elif problem_name == 'leaflet_contact':
            result = Procedure.run_procedure(self=self.problem, params=parameters)
            objective_values = result['objectives']
            objectives_dict = {
                'LMN_open': 1 - objective_values['LMN_open'],
                "LMN_closed": objective_values['LMN_closed'],
                "Smax": objective_values['Smax']
                # "I":  objective_values['I']
            }
            constraint_values = result['constraints']
            constraints_dict = {
                # "LMN_op_constr": constraint_values['LMN_op_constr'],
                # "LMN_cl_constr": constraint_values['LMN_cl_constr'],
                "VMS-Smax": constraint_values['VMS_constr'] - get_s_lim()
            }
        elif problem_name == 'test':
            curr_rand = random() * 100
            if curr_rand > get_percent():
                result = problem.evaluate(parameters)
                objective_values = result[0]
                objectives_dict = {
                    "objective1": objective_values[0],
                    "objective2": objective_values[1]
                }
                constraint_values = result[1]
                constraints_dict = {
                    "constraint1": constraint_values[0],
                    "constraint2": constraint_values[1],
                    "constraint3": constraint_values[2],
                    "constraint4": constraint_values[3]
                }
            else:
                # param_array = np.array(list(parameters.values()))
                # result = problem.evaluate(param_array)
                # objective_values = result[0]
                # objectives_dict = {
                #     "objective1": objective_values[0],
                #     "objective2": objective_values[1]
                # }
                objectives_dict = {
                    "objective1": 1000,
                    "objective2": 1000
                }
                # constraint_values = result[1]
                # constraints_dict = {
                #     "constraint1": constraint_values[0],
                #     "constraint2": constraint_values[1],
                #     "constraint3": constraint_values[2],
                #     "constraint4": constraint_values[3]
                # }
                constraints_dict = {
                    "constraint1": 100,
                    "constraint2": 100,
                    "constraint3": 100,
                    "constraint4": 100
                }
                print('value losted')

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
    try:
        X = pd.DataFrame(res.X, columns=problem.param_names)
    except:
        col = problem.param_names
        x_data = res.X
        res_x = dict()
        for valc, val in zip(col, x_data):
            res_x.update({valc: val})
        X = pd.DataFrame([res_x])
    F = pd.DataFrame(res.F, columns=problem.obj_names)

    try:
        G = pd.DataFrame() if not hasattr(res, 'G') or res.G is None else pd.DataFrame(res.G,
                                                                                       columns=problem.constr_names)
    except:
        if problem.constr_names == []:
            G = pd.DataFrame(res.G, columns=['no constr'])

    CV = pd.DataFrame() if not hasattr(res, 'CV') or res.CV is None else pd.DataFrame(res.CV, columns=['CV'])
    # opt_df = None
    # if res.opt is not None:
    #     opt_data = [{'X': dict(zip(problem.param_names, ind.X)),
    #                  'F': dict(zip(problem.obj_names, ind.F)),
    #                  'G': dict(zip(problem.constr_names, ind.G)) if ind.has("G") else None,
    #                  'CV': ind.CV if ind.has("CV") else None} for ind in res.opt]
    #     opt_df = pd.DataFrame(opt_data)

    pop_df = None
    if res.pop is not None:
        pop_data = [{'X': dict(zip(problem.param_names, ind.X)),
                     'F': dict(zip(problem.obj_names, ind.F)),
                     'G': dict(zip(problem.constr_names, ind.G)) if ind.has("G") else None,
                     'CV': ind.CV if ind.has("CV") else None} for ind in res.pop]
        pop_df = pd.DataFrame(pop_data)

    # return (history_df, X, F, G, CV, opt_df, pop_df)
    return history_df, X, F, G, CV, pop_df


def print_algorithm_params(
        algorithm_params: dict
) -> dict:
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


def print_termination_params(
        termination_params: dict
) -> None:
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


def create_results_folder(
        base_folder: str = 'results'
) -> str:
    """
    Creates a subfolder within a specified base folder to store results.
    The subfolder name is based on the current date and an incrementing number.

    Args:
        base_folder (str): The base folder where subfolders will be created. Defaults to 'results'.

    Returns:
        str: The path to the created subfolder.
    """
    if not os.path.exists(base_folder):
        os.makedirs(base_folder)

    existing_subfolders = [f for f in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, f))]

    subfolder_number = len(existing_subfolders) + 1
    current_time = datetime.datetime.now()
    formatted_date = current_time.strftime('%d_%m_%Y')

    subfolder_name = f"{subfolder_number:03}_{formatted_date}"
    folder_path = os.path.join(base_folder, subfolder_name)

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    print(f"Created folder: {folder_path}")

    return folder_path


def find_best_result(
        F: pd.DataFrame,
        weights: list[float]
) -> int:
    """
    Finds the best result in a multi-objective optimization based on ASF.

    Args:
        F (pd.DataFrame): DataFrame containing the objective values.
        weights (list[float]): A list of weights to assign to each objective.

    Returns:
        int: The index of the optimal solution in the DataFrame.
    """
    if not np.isclose(sum(weights), 1):  # Check if the sum of weights is approximately 1
        raise ValueError("Weights must sum to 1.")

    approx_ideal = F.min(axis=0)
    approx_nadir = F.max(axis=0)
    nF = (F - approx_ideal) / (approx_nadir - approx_ideal)
    nFm = nF.to_numpy(copy=True)
    decomp = ASF()
    best_index = decomp.do(nFm, 1 / np.array(weights)).argmin()

    return best_index


def save_optimization_summary(
        type_of_run: str = None,
        folder_path: str = None,
        best_index: int = None,
        elapsed_time: float = None,
        F: pd.DataFrame = None,
        X: pd.DataFrame = None,
        G: pd.DataFrame = None,
        history_df: pd.DataFrame = None,
        termination_params: dict = None,
        detailed_algo_params: dict = None
) -> None:
    """
    Save optimization summary to an Excel file with given details,
    including termination and detailed algorithm parameters.

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
        if type_of_run is None:
            headers = (["Timestamp", "Generation", "Best Index", "Elapsed Time"] + \
                       [f"F_{col}" for col in F.columns] + \
                       [f"X_{col}" for col in X.columns] + \
                       [f"G_{col}" for col in G.columns] + \
                       [f"Term_{key}" for key in termination_params] + \
                       headers_algo_params + \
                       ['Folder_path'])
        else:
            headers = (["Timestamp", 'Type of run', "Generation", "Best Index", "Elapsed Time"] + \
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
    if type_of_run is None:
        new_row = [timestamp, generation_number, best_index, elapsed_time] + \
                  best_F + best_X + best_G + termination_values + detailed_algo_values + [folder_path]
    else:
        new_row = [timestamp, type_of_run, generation_number, best_index, elapsed_time] + \
                  best_F + best_X + best_G + termination_values + detailed_algo_values + [folder_path]

    worksheet.append(new_row)
    workbook.save(summary_file)


def save_object(obj, filepath, filename):
    with open(os.path.join(folder_path, filename), 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
#    gc.enable()  # enable garbage collector to avoid memory leak
    basic_stdout = sys.stdout
    basic_stderr = sys.stderr
    set_mesh_step(0.6)
    # allowed 'test', 'beam', 'leaflet_single', 'leaflet_contact'
    problem_name = 'leaflet_single'
    set_dead_objects(0)
    pop_size = 80
    offsprings = 80
    crossover_chance = 0.9
    mutation_chance = 0.3
    set_problem_name(problem_name)
    crossover_eta = 50
    mutation_eta = 100

    percent = 0  # synthetic % of lost results

    if problem_name.lower() == 'test':
        typeof = f'Both loss {percent}%'
    elif problem_name.lower() == 'beam':
        set_base_name('beam_test')
        set_s_lim(1500)
        set_cpus(6)
        typeof = 'Abq beam'
    elif problem_name.lower() == 'leaflet_single':
        set_base_name('single_test')
        set_s_lim(3.3)
        set_cpus(6)
        typeof = 'Single leaf'
    elif problem_name.lower() == 'leaflet_contact':
        set_base_name('cont_test')
        set_s_lim(1.5)
        set_cpus(6)
        typeof = 'Contact'
    else:
        typeof = problem_name

    set_percent(percent)
    for file in glob('./*.rpy*'):
        os.remove(file)

    # folder to store results
    folder_path = create_results_folder(base_folder='results')

    # logging
    logger = setup_logger(folder_path)
    start_time = time.time()
    logger.info("Starting optimization...")

    set_id(0)
    if problem_name.lower() == 'beam':
        # parameter boundaries (min, max)
        parameters = {
            'Width': (1.0, 20.0),
            'THK': (1.0, 20.0)
        }
        # objectives names
        objectives = ['Displacement', 'Mass']
        # constraints names
        constraints = ['THK_constr', 'Width_constr', 'Smax_constr']
        ref_point = np.array(
            [
                (max(parameters['THK']) - max(parameters['THK'])) / 2,
                (max(parameters['Width']) - max(parameters['Width'])) / 2,
                get_s_lim() / 2
            ]
        )
    elif problem_name.lower() == 'test':
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
        ref_point = np.array([100, 0.1])
    elif problem_name.lower() == 'leaflet_single' or problem_name.lower() == 'leaflet_contact':
        parameters = {
            'Lstr': (0, 1),
            'ANG': (-10, 10),
            'CVT': (0.1, 0.8),
            'LAS': (0.2, 1.5)
        }
        # objectives = ['LMN_open', 'LMN_closed', 'Smax']
        objectives = ['1 - LMN_open', 'Smax']
        constraints = [  # 'LMN_op_constr'],
        # 'LMN_cl_constr',
        'VMS-Smax']
        ref_point = np.array([1, 0, get_s_lim()])
    print('Parameters:', parameters)
    print('Objectives:', objectives)
    print('Constraints:', constraints)

    # problem initialization
    problem = Problem(parameters, objectives, constraints)

    # algorithm initialization
    algorithm_params = {
        "pop_size": pop_size,
        "n_offsprings": offsprings,
        "sampling": FloatRandomSampling(),
        "crossover": SBX(prob=crossover_chance, eta=crossover_eta),
        "mutation": PM(prob=mutation_chance, eta=mutation_eta),
        "eliminate_duplicates": True
    }
    algorithm = NSGA2(**algorithm_params)
    detailed_algo_params = print_algorithm_params(algorithm_params)

    # termination criteria
    termination_params = {
        "xtol": 1e-8,
        "cvtol": 1e-6,
        "ftol": 0.0025,
        "period": 5,
        "n_max_gen": 1000,
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

    # save_object(res, folder_path, 'results_object.pkl')

    try:
        # result storage
        history_df, X, F, G, CV, pop = extract_optimization_results(res, problem, folder_path)
        history_df.to_csv(os.path.join(folder_path, 'history.csv'))
        X.to_csv(os.path.join(folder_path, 'X.csv'))
        F.to_csv(os.path.join(folder_path, 'F.csv'))
        G.to_csv(os.path.join(folder_path, 'G.csv'))
        CV.to_csv(os.path.join(folder_path, 'CV.csv'))
        pop.to_csv(os.path.join(folder_path, 'pop.csv'))

        #  Find the best trade-off between objectives using Augmented Scalarization Function (ASF)

        # weights = [0.5, 0.5]
        weights = np.zeros(len(objectives))
        for v in range(len(weights)):
            weights[v] = 1 / len(objectives)

        best_index = find_best_result(F, list(weights))
        print(f'Best regarding ASF:\nPoint #{best_index}\n{F.iloc[best_index]}')
        print('Elapsed time:', elapsed_time)
        logger.info("Optimization completed.")

        # upload result to integrate table
        save_optimization_summary(
            typeof,
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
    except Exception as e:
        print(f'Exception {e}')

    csv_files = {
        'history': 'history.csv',
        'X': 'X.csv',
        'F': 'F.csv',
        'G': 'G.csv',
        'CV': 'CV.csv',
        'pop': 'pop.csv'
    }
    optimization_results = load_optimization_results(folder_path, csv_files)

    # Best trade-off between objectives using ASF
    try:
        plot_best_objectives(F=optimization_results['F'], weights='equal', folder_path=folder_path)
        print(colored("Best trade-off plot created successfully.", "green"))
    except Exception as e:
        print(colored(f"Failed to plot best trade-off objectives: {str(e)}", "red"))

    # Objectives vs parameters
    try:
        plot_objectives_vs_parameters(
            X=optimization_results['X'],
            F=optimization_results['F'],
            folder_path=folder_path
        )
        print(colored("Objectives vs Parameters plotted successfully.", "green"))
    except Exception as e:
        print(colored(f"Failed to plot Objectives vs Parameters: {str(e)}", "red"))

    # Constrains vs parameters
    try:
        plot_constrains_vs_parameters(
            X=optimization_results['X'],
            G=optimization_results['G'],
            folder_path=folder_path
        )
        print(colored("Constrains vs Parameters plotted successfully.", "green"))
    except Exception as e:
        print(colored(f"Failed to plot Constrains vs Parameters: {str(e)}", "red"))

    # Convergence for objectives
    try:
        plot_objective_convergence(optimization_results['history'], objectives,
                                   folder_path)
        print(colored("Objective Convergence plotted successfully.", "green"))
    except Exception as e:
        print(colored(f"Failed to plot Objective Convergence: {str(e)}", "red"))

    # Parallel coordinates plot
    try:
        plot_parallel_coordinates(
            X=optimization_results['X'],
            G=optimization_results['G'],
            F=optimization_results['F'],
            objectives=objectives,
            folder_path=folder_path
        )
        print(colored("Parallel Coordinates plotted successfully.", "green"))
    except Exception as e:
        print(colored(f"Failed to plot Parallel Coordinates: {str(e)}", "red"))

    # Convergence by Hypervolume
    try:
        plot_convergence_by_hypervolume(optimization_results['history'], objectives, folder_path)
        print(colored("Convergence by Hypervolume plotted successfully.", "green"))
    except Exception as e:
        print(colored(f"Failed to plot Convergence by Hypervolume: {str(e)}", "red"))

    # Parameters over generation
    try:
        plot_parameters_over_generation(
            optimization_results['history'],
            problem.param_names,
            problem.obj_names,
            folder_path
        )
        print(colored("Parameters over generations plotted successfully.", "green"))
    except Exception as e:
        print(colored(f"Failed to create Parameters over generations  plot: {str(e)}", "red"))

    # Pareto front
    try:
        plot_pareto_with_trade_off(
            history=optimization_results['history'],
            F=optimization_results['F'],
            objectives=problem.obj_names,
            weights='equal',
            folder_path=folder_path
        )
        print(colored("Best trade-off plot created successfully.", "green"))
    except Exception as e:
        print(colored(f"Failed to plot best trade-off objectives: {str(e)}", "red"))
    cleanup_logger(logger)
    del logger
    sys.stdout = basic_stdout
    sys.stderr = basic_stderr
