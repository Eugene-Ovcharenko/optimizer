import os
import sys
import datetime
import pathlib
import logging
import openpyxl
import pandas as pd
from glob2 import glob

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

    project_folder = "/".join(str(pathlib.Path(__file__).parent.resolve()).split("/")[:-1])

    subfolder_name = f"{subfolder_number:03}_{formatted_date}"
    basic_folder_path = os.path.join(project_folder, base_folder, subfolder_name)

    if not os.path.exists(basic_folder_path):
        os.makedirs(basic_folder_path)

    print(f"Created folder: {basic_folder_path}")

    return basic_folder_path


def setup_logger(basic_folder_path: str, log_file_name: str = 'terminal_log.txt') -> logging.Logger:
    #Sets up a logger to capture verbose output, writing to both a log file and the console.
    #Args: basic_folder_path (str): The directory path to save the log file.
    #log_file_name (str, optional): The name of the log file. Defaults to 'terminal_log.txt'.
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
):
    base_folder = '/'.join(folder_path.split(os.path.sep)[:-1])
    summary_file = os.path.join(base_folder, 'optimization_summary.xlsx')

    if os.path.exists(summary_file):
        workbook = openpyxl.load_workbook(summary_file)
        worksheet = workbook.active
    else:
        workbook = openpyxl.Workbook()
        worksheet = workbook.active
        headers = (["Timestamp", "Type of run", "Generation", "Best Index", "Elapsed Time"] +
                   [f"F_{col}" for col in F.columns] +
                   [f"X_{col}" for col in X.columns] +
                   [f"G_{col}" for col in G.columns] +
                   [f"Term_{key}" for key in termination_params] +
                   list(detailed_algo_params.keys()) +
                   ['Folder_path'])
        worksheet.append(headers)

    generation_number = history_df['generation'].max()
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    best_F = F.iloc[best_index].tolist()
    best_X = X.iloc[best_index].tolist()
    best_G = G.iloc[best_index].tolist()

    termination_values = list(termination_params.values())
    detailed_algo_values = list(detailed_algo_params.values())

    new_row = [timestamp, type_of_run, generation_number, best_index, elapsed_time] + \
              best_F + best_X + best_G + termination_values + detailed_algo_values + [folder_path]

    worksheet.append(new_row)
    workbook.save(summary_file)


def purgeFiles(
    endPath: str,
    partName: str,
    pathToAbaqus: str,
    jobName: str
) -> None:
    """
    Remove intermediate and redundant files generated by Abaqus simulations to maintain a clean working directory.

    This function:
    - Deletes the result directory associated with the simulation part.
    - Removes auxiliary files (e.g., `.simdir`, temporary files) while preserving key outputs such as `.odb`, `.py`, and `.txt`.
    - Cleans both Windows and Unix-based environments via platform-specific system calls.
    - Ensures the workspace is ready for the next simulation run, which is essential in automated pipelines.

    Args:
        endPath (str): Directory path where the part-specific results are stored.
        partName (str): Name of the part or model, used to construct the directory name.
        pathToAbaqus (str): Path to the directory where the Abaqus job files are located.
        jobName (str): Base name of the Abaqus simulation job (used to locate relevant files).

    Returns:
        None

    Notes:
        - On Windows, `rmdir` and `del` are used. On Unix-like systems, `rm -r` is used.
        - All files beginning with the job name are scanned. Files with `.py`, `.odb`, or `.txt` extensions are preserved.
        - Additional files matching the `explicit*` pattern are also deleted.
        - Exceptions during deletion are silently ignored, which may conceal permission or I/O issues.
        - This function assumes proper write access to the specified directories.

    Example:
        >>> purgeFiles("C:/SimResults/", "valve_model.inp", "C:/AbaqusJobs/", "valve_model")
    """

    if str.upper(sys.platform) == 'WIN32':
        rm_com = 'del '
        os.system('rmdir /s /q ' + endPath + partName[0:-5].upper())
        os.system(pathToAbaqus + jobName + ".sim* ")
    else:
        rm_com = 'rm -r '
        os.system(rm_com + endPath + partName[0:-5].upper())
        if os.path.exists(pathToAbaqus + jobName + ".simdir"):
            os.system(rm_com + pathToAbaqus + jobName + ".simdir")

    # Get all files matching the pattern
    all_files = glob(os.path.join(pathToAbaqus, jobName) + '*')

    # Filter out files with extensions you want to exclude
    files_to_delete = [
        f for f in all_files
        if not (f.endswith(".py") or f.endswith(".odb") or f.endswith(".txt"))
    ]

    # Delete the filtered files
    for file_path in files_to_delete:
        try:
            os.remove(file_path)
        except Exception as e:
            pass
    all_files = glob(os.path.join(pathToAbaqus, 'explicit*'))
    for file_path in all_files:
        try:
            os.remove(file_path)
        except Exception as e:
            pass