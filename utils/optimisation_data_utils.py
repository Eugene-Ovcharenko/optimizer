import datetime

import pandas as pd
import openpyxl

from pymoo.core.problem import ElementwiseProblem
from pymoo.core.result import Result

from pymoo.operators.sampling.rnd import FloatRandomSampling

from utils.visualize import *


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

    if res.X:
        X = pd.DataFrame(res.X, columns=problem.param_names)
    else:
        X = pd.DataFrame(history_df[problem.param_names], columns=problem.param_names)

    if res.F:
        F = pd.DataFrame(res.F, columns=problem.obj_names)
    else:
        F = pd.DataFrame(history_df[problem.obj_names], columns=problem.obj_names)


    if hasattr(res, 'G'):
        if res.G:
            G = pd.DataFrame(res.G, columns=problem.constr_names)
        else:
            G = pd.DataFrame(history_df[problem.constr_names], columns=problem.constr_names)
    else:
        G = pd.DataFrame(res.G, columns=['no constr'])


    if hasattr(res, 'CV'):
        if res.CV:
            CV = pd.DataFrame(res.CV, columns=['CV'])
        else:
            CV = pd.DataFrame(history_df['CV'], columns=['CV'])
        # CV = pd.DataFrame(res.CV, columns=['CV'])
    else:
        CV = pd.DataFrame([], columns=['CV'])

    pop_df = None
    if res.pop is not None:
        pop_data = [{'X': dict(zip(problem.param_names, ind.X)),
                     'F': dict(zip(problem.obj_names, ind.F)),
                     'G': dict(zip(problem.constr_names, ind.G)) if ind.has("G") else None,
                     'CV': ind.CV if ind.has("CV") else None} for ind in res.pop]
        pop_df = pd.DataFrame(pop_data)

    return history_df, X, F, G, CV, pop_df
