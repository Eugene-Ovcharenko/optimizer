import sys
import os
import datetime
import logging
import pickle
import time
from typing import Tuple, Optional, List, Dict

import pandas as pd
from glob2 import glob
import openpyxl
import pathlib

import numpy as np
from omegaconf import DictConfig
import hydra
import seaborn as sns

from pymoo.algorithms.moo.moead import MOEAD
from pymoo.core.callback import Callback
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.result import Result
from pymoo.decomposition.asf import ASF
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize
from pymoo.termination.default import DefaultMultiObjectiveTermination

from utils.global_variable import *
from utils.problem_no_xlsx import Procedure, init_procedure
from utils.visualize import *

config_name = 'config_leaf_MOEAD'


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


def setup_logger(basic_folder_path: str, log_file_name: str = 'terminal_log.txt') -> logging.Logger:
    # Sets up a logger to capture verbose output, writing to both a log file and the console.
    # Args: basic_folder_path (str): The directory path to save the log file.
    # log_file_name (str, optional): The name of the log file. Defaults to 'terminal_log.txt'.
    # Returns: logging.Logger: Configured logger for capturing terminal output.  """ 
    if not os.path.exists(basic_folder_path):
        os.makedirs(basic_folder_path)

    logger = logging.getLogger('multi_stream_logger')
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(os.path.join(basic_folder_path, log_file_name))
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

        def eval_safe(expr: str, vars_dict: dict):
            expr = expr.replace('^', '**')
            # запрещаем доступ к __builtins__
            safe_globals = {"__builtins__": None}
            # разрешаем только переменные из vars_dict
            return eval(expr, safe_globals, vars_dict)

        params = dict(zip(self.param_names, x))
        # result = Procedure.run_procedure(self=problem, params=x)
        self.problem = init_procedure(np.array(x))
        problem_name = get_problem_name().lower()
        cpus = get_cpus()
        parameters = np.array(x)
        constraints_dict = dict()

        if problem_name == 'leaflet_single':
            result = Procedure.run_procedure(self=self.problem, params=parameters)
            objective_values = result['results']

            vars_dict = {
                'LMN_open': objective_values['LMN_open'],
                'LMN_closed': objective_values['LMN_closed'],
                'Slim': get_s_lim(),
                'Smax': objective_values['Smax'],
                'HELI': objective_values['HELI']
            }

        elif problem_name == 'leaflet_contact':
            result = Procedure.run_procedure(self=self.problem, params=parameters)
            objective_values = result['results']
            # --- 3. Строим словарь переменных для eval
            vars_dict = {
                'LMN_open': objective_values['LMN_open'],
                'LMN_closed': objective_values['LMN_closed'],
                'Slim': get_s_lim(),
                'Smax': objective_values['Smax'],
                'HELI': objective_values['HELI']
            }
        elif problem_name == 'test':
            result = Procedure.run_procedure(self=self.problem, params=parameters)
            objective_values = result['objectives']
            vars_dict = {
                "objective1": objective_values['objective1'],
                "objective2": objective_values['objective2']
            }
            constraint_values = result['constraints']
            constraints_dict = {
                "constraint1": constraint_values["constraint1"],
                "constraint2": constraint_values["constraint2"],
                "constraint3": constraint_values["constraint3"],
                "constraint4": constraint_values["constraint4"]
            }

        objectives_vars = [eval_safe(expr, vars_dict) for expr in self.obj_names]
        if constraints_dict == {}:
            constraints_vars = [eval_safe(expr, vars_dict) for expr in self.constr_names]

        objectives_dict = dict(zip(self.obj_names, objectives_vars))
        constraints_dict = dict(zip(self.constr_names, constraints_vars))

        result = {
            'objectives': objectives_dict,
            'constraints': constraints_dict
        }

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
        if not problem.constr_names:
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
    basic_folder_path = os.path.join(pathlib.Path(__file__).parent.resolve(), base_folder, subfolder_name)

    if not os.path.exists(basic_folder_path):
        os.makedirs(basic_folder_path)

    print(f"Created folder: {basic_folder_path}")

    return basic_folder_path


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
    base_folder = '/'.join(folder_path.split(os.path.sep)[:-1])
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
            headers = (["Timestamp", "Generation", "Best Index", "Elapsed Time"] +
                       [f"F_{col}" for col in F.columns] +
                       [f"X_{col}" for col in X.columns] +
                       [f"G_{col}" for col in G.columns] +
                       [f"Term_{key}" for key in termination_params] +
                       headers_algo_params +
                       ['Folder_path'])
        else:
            headers = (["Timestamp", 'Type of run', "Generation", "Best Index", "Elapsed Time"] +
                       [f"F_{col}" for col in F.columns] +
                       [f"X_{col}" for col in X.columns] +
                       [f"G_{col}" for col in G.columns] +
                       [f"Term_{key}" for key in termination_params] +
                       headers_algo_params +
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


def load_state(filepath='./', filename='checkpoint.pkl'):
    # return joblib.load(os.path.join(filepath,filename))
    with open(os.path.join(filepath, filename), "rb") as f:
        return pickle.load(f)


# callback class for saving
class CustomCallback(Callback):
    def __init__(self, objectives=['a', 'b'], folder_path='./utils/logs/', interval_backup=10, interval_picture=2):
        print('Callback inited')
        super().__init__()
        self.objectives = objectives
        self.folder_path = folder_path
        self.interval_backup = interval_backup
        self.interval_picture = interval_picture
        self.min_values = []

    def save_state(self, algorithm):
        with open(os.path.join(self.folder_path, f'checkpoint_{algorithm.n_gen:03d}.pkl'), 'wb') as outp:
            state = {
                "X": algorithm.pop.get("X"),
                "F": algorithm.pop.get("F"),
                "n_gen": algorithm.n_gen,
                "algorithm_state": {
                    key: value for key, value in algorithm.__dict__.items()
                    if not isinstance(value, (openpyxl.Workbook, openpyxl.worksheet.worksheet.Worksheet))
                }
            }
            # joblib.dump(state, self.filename)
            pickle.dump(state, outp)
            print(f'State saved. {int(algorithm.n_gen):03d}')

    def plot_convergence(self, algorithm):
        # Получаем все значения целевых функций в текущей популяции
        # Предполагается, что F - это массив значений целевых функций для каждого индивида
        objectives = [ind.F for ind in algorithm.pop]
        # Преобразуем в DataFrame для удобства обработки
        objectives_df = pd.DataFrame(objectives, columns=self.objectives)
        # Находим минимальные значения по каждой целевой функции
        min_objectives = objectives_df.min()
        # Находим минимальное значение среди всех целевых функций
        overall_min = min_objectives.min()
        # Добавляем текущую генерацию и минимальное значение в историю
        self.min_values.append((algorithm.n_gen, min_objectives))
        num_objectives = len(min_objectives)
        # Построение графика с использованием Seaborn
        sns.set(style="whitegrid")
        fig, axes = plt.subplots(1, num_objectives, figsize=(15, 5), sharey=False)
        if num_objectives == 1:
            axes = [axes]
        gens, vals = zip(*self.min_values)
        for idx, obj_col in enumerate(self.objectives):
            y_values = [val[obj_col] for val in vals]
            sns.lineplot(x=gens, y=y_values, ax=axes[idx],
                         marker='o', color='b', markeredgecolor=None, label=f"Best {obj_col}")
            if len(y_values) > 2:
                axes[idx].set_title(f"Convergence of ({obj_col}) | Δ={abs(y_values[-2] - y_values[-1]):2.2e}")
            else:
                axes[idx].set_title(f"Convergence of ({obj_col})")
            axes[idx].set_xlabel("Generation")
        # Сохранение графика с указанием текущей генерации
        # plt.savefig(os.path.join(self.folder_path, f'intime_convergence_gen_{algorithm.n_gen}.png'))
        plt.savefig(os.path.join(self.folder_path, f'intime_convergence.png'))
        plt.close()
        # time.sleep(2)

    def notify(self, algorithm):
        if algorithm.n_gen % self.interval_backup == 0:
            self.save_state(algorithm)
        if algorithm.n_gen % self.interval_picture == 0:
            self.plot_convergence(algorithm)


@hydra.main(config_path="configuration", config_name=config_name, version_base=None)
def main(cfg: DictConfig):
    basic_stdout = sys.stdout
    basic_stderr = sys.stderr

    parameters = {k: tuple(v) for k, v in cfg.parameters.items()}
    objectives = cfg.objectives

    print("\nConverted parameters (as tuples):")
    print(parameters)
    print("\nObjectives:")
    print(objectives)

    set_cpus(cfg.Abaqus.abq_cpus)
    set_tangent_behavior(cfg.Abaqus.tangent_behavior)
    set_normal_behavior(cfg.Abaqus.normal_behavior)

    set_DIA(cfg.problem_definition.DIA)
    set_Lift(cfg.problem_definition.Lift)
    set_SEC(cfg.problem_definition.SEC)
    set_EM(cfg.problem_definition.EM)
    set_density(cfg.problem_definition.Dens)
    set_material_name(cfg.problem_definition.material_name)
    set_mesh_step(cfg.problem_definition.mesh_step)
    set_valve_position(cfg.problem_definition.position)
    set_problem_name(cfg.problem_definition.problem_name)
    set_base_name(cfg.problem_definition.problem_name)
    set_s_lim(cfg.problem_definition.s_lim)
    set_global_path(str(pathlib.Path(__file__).parent.resolve()))

    set_mesh_step(0.4)
    set_valve_position('mitr')  # can be 'mitr'
    problem_name = get_problem_name()

    percent = 0  # synthetic % of lost results

    restore_state = False

    # folder to store results
    if not restore_state:
        basic_folder_path = create_results_folder(base_folder='results')
    else:
        basic_folder_path = os.path.join(pathlib.Path(__file__).parent.resolve(), 'results', '007_14_01_2025')
        last_gen = 10

    print(f"folder path > {basic_folder_path}")

    if problem_name.lower() == 'test':
        typeof = f'Both loss {percent}%'
    elif problem_name.lower() == 'leaflet_single':
        set_base_name('single_test')
        set_s_lim(3.3)
        set_cpus(6)
        typeof = 'Single leaf'
    elif problem_name.lower() == 'leaflet_contact':
        set_base_name('Mitral_test')
        set_s_lim(3.23)  # Formlabs elastic 50A
        set_cpus(3)  # 3 cpu cores shows better results then 8 cores. 260sec vs 531sec
        typeof = 'Contact'
    else:
        typeof = problem_name

    set_percent(percent)
    for file in glob('./*.rpy*'):
        os.remove(file)

    # logging
    logger = setup_logger(basic_folder_path)

    start_time = time.time()
    logger.info("Starting optimization...")

    set_id(0)
    if problem_name.lower() == 'test':
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
        # constraints = ['constraint1', 'constraint2', 'constraint3', 'constraint4']
        constraints = []
        ref_point = np.array([100, 0.1])
    elif problem_name.lower() == 'leaflet_single' or problem_name.lower() == 'leaflet_contact':
        parameters = {
            'HGT': (10, 15.5),
            'Lstr': (0, 1),
            'THK': (0.25, 0.6),
            'ANG': (-10, 10),
            'CVT': (0.1, 0.8),
            'LAS': (0.2, 1.5)
        }
        objectives = ['1 - LMN_open', 'LMN_closed', 'Smax - Slim', 'HELI', 'Gauss_curv']
        constraints = []
        ref_point = np.array([1, 0, get_s_lim(), 0])
    print('Parameters:', parameters)
    print('Objectives:', objectives)
    print('Constraints:', constraints)

    # problem initialization
    problem = Problem(parameters, objectives, constraints)

    # callback initialisation
    save_callback = CustomCallback(
        objectives=objectives,
        folder_path=basic_folder_path,
        interval_backup=10,
        interval_picture=1
    )

    ref_dirs = get_reference_directions("uniform", len(objectives), n_partitions=cfg.optimizer.pop_size)
    # algorithm initialization
    algorithm_params = {
        "ref_dirs": ref_dirs,
        "n_offsprings": cfg.optimizer.offsprings,
        "sampling": FloatRandomSampling(),
        "crossover": SBX(prob=cfg.optimizer.crossover_chance, eta=cfg.optimizer.crossover_eta),
        "mutation": PM(prob=cfg.optimizer.mutation_chance, eta=cfg.optimizer.mutation_eta),
        "neighborhood_size": cfg.optimizer.neighborhood_size,
        "prob_neighbor_mating": cfg.optimizer.prob_neighbor_mating,
        "decomposition": ASF()  # or any other decomposition
    }
    algorithm = MOEAD(**algorithm_params)
    detailed_algo_params = print_algorithm_params(algorithm_params)

    # termination criteria
    termination_params = {
        "xtol": cfg.optimizer.termination_parameters.xtol,
        "cvtol": cfg.optimizer.termination_parameters.cvtol,
        "ftol": cfg.optimizer.termination_parameters.ftol,
        "period": cfg.optimizer.termination_parameters.period,
        "n_max_gen": cfg.optimizer.termination_parameters.n_max_gen,
        "n_max_evals": cfg.optimizer.termination_parameters.n_max_evals
    }
    termination = DefaultMultiObjectiveTermination(**termination_params)
    print_termination_params(termination_params)

    if restore_state:
        try:
            state = load_state(filepath=basic_folder_path, filename=f'checkpoint_{last_gen:03d}.pkl')
            algorithm.n_gen = state["n_gen"]
            for key, value in state["algorithm_state"].items():
                setattr(algorithm, key, value)
            print(f'State was loaded. Current gen is {algorithm.n_gen}')
        except FileNotFoundError:
            print(f'Pickle file not found!')
            sys.exit(1)

    # run optimization
    res = minimize(problem,
                   algorithm,
                   termination,
                   seed=1,
                   save_history=True,
                   callback=save_callback,
                   verbose=True)
    elapsed_time = time.time() - start_time

    try:
        # result storage
        history_df, X, F, G, CV, pop = extract_optimization_results(res, problem, basic_folder_path)
        history_df.to_csv(os.path.join(basic_folder_path, 'history.csv'))
        X.to_csv(os.path.join(basic_folder_path, 'X.csv'))
        F.to_csv(os.path.join(basic_folder_path, 'F.csv'))
        G.to_csv(os.path.join(basic_folder_path, 'G.csv'))
        CV.to_csv(os.path.join(basic_folder_path, 'CV.csv'))
        pop.to_csv(os.path.join(basic_folder_path, 'pop.csv'))

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
            basic_folder_path,
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
    optimization_results = load_optimization_results(basic_folder_path, csv_files)

    # Best trade-off between objectives using ASF
    try:
        plot_best_objectives(F=optimization_results['F'], weights='equal', folder_path=basic_folder_path)
        print(colored("Best trade-off plot created successfully.", "green"))
    except Exception as e:
        print(colored(f"Failed to plot best trade-off objectives: {str(e)}", "red"))

    # Objectives vs parameters
    try:
        plot_objectives_vs_parameters(
            X=optimization_results['X'],
            F=optimization_results['F'],
            folder_path=basic_folder_path
        )
        print(colored("Objectives vs Parameters plotted successfully.", "green"))
    except Exception as e:
        print(colored(f"Failed to plot Objectives vs Parameters: {str(e)}", "red"))

    # Constrains vs parameters
    try:
        plot_constrains_vs_parameters(
            X=optimization_results['X'],
            G=optimization_results['G'],
            folder_path=basic_folder_path
        )
        print(colored("Constrains vs Parameters plotted successfully.", "green"))
    except Exception as e:
        print(colored(f"Failed to plot Constrains vs Parameters: {str(e)}", "red"))

    # Convergence for objectives
    try:
        plot_objective_convergence(optimization_results['history'], objectives,
                                   basic_folder_path)
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
            folder_path=basic_folder_path
        )
        print(colored("Parallel Coordinates plotted successfully.", "green"))
    except Exception as e:
        print(colored(f"Failed to plot Parallel Coordinates: {str(e)}", "red"))

    # Convergence by Hypervolume
    try:
        plot_convergence_by_hypervolume(optimization_results['history'], objectives, basic_folder_path)
        print(colored("Convergence by Hypervolume plotted successfully.", "green"))
    except Exception as e:
        print(colored(f"Failed to plot Convergence by Hypervolume: {str(e)}", "red"))

    # Parameters over generation
    try:
        plot_parameters_over_generation(
            optimization_results['history'],
            problem.param_names,
            problem.obj_names,
            basic_folder_path
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
            folder_path=basic_folder_path
        )
        print(colored("Best trade-off plot created successfully.", "green"))
    except Exception as e:
        print(colored(f"Failed to plot best trade-off objectives: {str(e)}", "red"))

    # Best leaflet - frame connection contour
    try:
        HGT, Lstr, THK, ANG, CVT, LAS = optimization_results['history'][parameters.keys()].iloc[best_index]
        DIA = 29 - 2 * 1.5
        Lift = 0
        SEC = 119

        from utils.create_geometry_utils import createGeometry

        pointsInner, _, _, _, pointsHullLower, _, points, _, finalRad, currRad, message = \
            createGeometry(HGT=HGT, Lstr=Lstr, SEC=SEC, DIA=DIA, THK=THK,
                           ANG=ANG, Lift=Lift, CVT=CVT, LAS=LAS, mesh_step=get_mesh_step())
        os.makedirs('inps', exist_ok=True)
        with open(f'{basic_folder_path}/fixed_bottom_{best_index}.txt', 'w') as writer:
            for point in pointsHullLower.T:
                writer.write("%6.6f %6.6f %6.6f\n" % (point[0], point[1], point[2]))
        print(colored("Best leaflet - frame connection contour created successfully.", "green"))
    except Exception as e:
        print(colored(f"Failed to create best leaflet - frame connection contour: {str(e)}", "red"))
    cleanup_logger(logger)
    del logger
    sys.stdout = basic_stdout
    sys.stderr = basic_stderr


if __name__ == "__main__":
    main()
