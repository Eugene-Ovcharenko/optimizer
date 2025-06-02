import os
import datetime
import time
import subprocess
import psutil as psu
from utils.global_variable import get_problem_name, get_cpus

def run_abaqus(
    Path: str = None,
    jobName: str = None,
    InpFile: str = None,
    cpus: int = None
) -> str:
    """
    Execute an Abaqus finite element analysis (FEA) simulation via the command line and monitor execution state.

    This function:
    - Launches Abaqus in multithreaded mode using the specified input file.
    - Checks for the `.lck` lock file to determine if the job is still running.
    - Periodically polls system processes to detect potential errors (e.g., stalled 'pre' or 'package' modules).
    - Forces termination if runtime exceeds 60 minutes or specific error conditions are detected.
    - Returns a completion or failure message based on observed status.

    Args:
        Path (str): Absolute path to the directory containing the `.inp` file and target working directory.
        jobName (str): Base name of the Abaqus job (used to construct `.inp` and `.lck` filenames).
        InpFile (str): Filename of the Abaqus input file (typically ends with `.inp`).
        cpus (int): Number of CPU threads to use during the analysis (parallel execution).

    Returns:
        str: Status message indicating whether Abaqus completed successfully, failed to launch,
             terminated due to time constraints, or was killed due to preprocessor/package errors.

    Raises:
        RuntimeError: If the working directory cannot be changed or if subprocess execution fails
                      (currently unhandled, but could be added for robustness).

    Notes:
        - Uses `subprocess.check_output()` to launch Abaqus; assumes it is on system PATH.
        - Requires the `psutil` package (`psu` alias) for process inspection.
        - Designed to support batch or automated job execution in high-throughput environments.
        - Logs execution progress using a custom `log_message(...)` function (assumed to be defined externally).
    """
    inputFile = 'abaqus job=' + str(jobName) + ' inp=' + str(InpFile) + ' cpus=' + str(
        cpus) + ' mp_mode=threads ask_delete=OFF'
    t0 = datetime.datetime.now()
    previous_path = os.getcwd()
    os.chdir(Path)
    out_args = subprocess.check_output(inputFile, shell=True)
    time.sleep(10)
    os.chdir(previous_path)
    message = 'ABAQUS complete'
    checked = False
    current_user = psu.users()[0].name
    if (os.path.exists(Path + '/' + jobName + '.lck')):
        while (os.path.exists(Path + '/' + jobName + '.lck')):
            t = datetime.datetime.now() - t0
            sec = t.seconds
            m = int(sec / 60) % 60
            h = int(sec / 3600)
            if m > 1 and not checked:
                for proc in psu.process_iter(['name', 'username']):
                    if 'pre' in proc.name() and current_user in proc.username():
                        os.system('pkill -n -9 pre')
                        message = 'ABAQUS terminated with error in pre'
                        break
                    if 'package' in proc.name() and current_user in proc.username():
                        os.system('pkill -n -9 package')
                        message = 'ABAQUS terminated with error in package'
                        break
                checked = True
            if m < 60:
                time.sleep(30)
            else:
                os.system('pkill -n -9 explicit')
                message = 'ABAQUS terminated due time'
                break
    else:
        message = 'runanaqus error: InpFile submit failed'
    return message

def get_history_output_single(
    pathName: str = None,
    odbFileName: str = None,
    cpus: int = -1
) -> None:
    """
    Trigger Abaqus CAE to execute a custom script for extracting history output from a `.odb` file.

    This function prepares a request file (`req.txt`) containing the path and name of the output database file.
    It then launches Abaqus CAE in non-GUI mode to run the embedded Python script `odbHistoryOutput_4perField.py`,
    which uses Abaqus' internal API to extract field-based history data (e.g., reaction forces, displacements)
    for single leaflet simulation.

    Args:
        pathName (str): Absolute or relative path to the Abaqus job folder containing `.odb` and script.
        odbFileName (str): Name of the Abaqus output database (`.odb`) to be processed.
        cpus (int): Number of CPUs to allocate (not used in this implementation; placeholder for extensibility).

    Returns:
        None

    Notes:
        - The script `odbHistoryOutput_4perField.py` must exist in the directory `pathName` and should not be deleted.
        - Two copies of `req.txt` are written (in both working directory and `pathName`) to conform to Abaqus script expectations.
        - The use of `abaqus cae noGUI=...` is necessary to invoke Abaqus-specific Python APIs that are not available in standard Python environments.
        - This function assumes that Abaqus is accessible from the system PATH and that the user has required execution privileges.

    Example:
        >>> get_history_output_single('./simulation_run/', 'valve_model.odb')
    """
    # prepare result folder
    if not os.path.exists(pathName + 'results/'):
        os.makedirs(pathName + 'results/')

    reqFile = str(pathName) + '/req.txt'
    fid = open(reqFile, 'wt')
    fid.write('%s,%s' % (pathName, odbFileName))
    fid.close()
    reqFile = './req.txt'
    fid = open(reqFile, 'wt')
    fid.write('%s,%s' % (pathName, odbFileName))
    fid.close()
    consoleCommand = 'abaqus cae noGUI=' + str(os.path.join(pathName,'abaqus_scripts/')) + 'odbHistoryOutput_4perField.py'
    outputs_args = os.system(consoleCommand)

def get_history_output_contact(
    pathName: str = None,
    odbFileName: str = None,
    cpus: int = -1
) -> None:
    """
    Trigger Abaqus CAE to execute a custom script for extracting history output from a `.odb` file.

    This function prepares a request file (`req.txt`) containing the path and name of the output database file.
    It then launches Abaqus CAE in non-GUI mode to run the embedded Python script `odbHistoryOutput_4perField.py`,
    which uses Abaqus' internal API to extract field-based history data (e.g., reaction forces, displacements)
    for 3-leaflet simulation with contacts.

    Args:
        pathName (str): Absolute or relative path to the Abaqus job folder containing `.odb` and script.
        odbFileName (str): Name of the Abaqus output database (`.odb`) to be processed.
        cpus (int): Number of CPUs to allocate (not used in this implementation; placeholder for extensibility).

    Returns:
        None

    Notes:
        - The script `odbHistoryOutput_4perField.py` must exist in the directory `pathName` and should not be deleted.
        - Two copies of `req.txt` are written (in both working directory and `pathName`) to conform to Abaqus script expectations.
        - The use of `abaqus cae noGUI=...` is necessary to invoke Abaqus-specific Python APIs that are not available in standard Python environments.
        - This function assumes that Abaqus is accessible from the system PATH and that the user has required execution privileges.

    Example:
        >>> get_history_output_single('./simulation_run/', 'valve_model.odb')
    """
    # prepare result folder
    if not os.path.exists(pathName + 'results/'):
        os.makedirs(pathName + 'results/')
    reqFile = str(pathName) + '/req.txt'
    fid = open(reqFile, 'wt')
    fid.write('%s,%s' % (pathName, odbFileName))
    fid.close()
    reqFile = './req.txt'
    fid = open(reqFile, 'wt')
    fid.write('%s,%s' % (pathName, odbFileName))
    fid.close()
    consoleCommand = 'abaqus cae noGUI=' + str(os.path.join(pathName,'abaqus_scripts/')) + 'odbHistoryOutput_ShellContact.py'
    outputs_args = os.system(consoleCommand)

def get_history_output(
    pathName: str = None,
    odbFileName: str = None,
    cpus: int = -1
) -> None:
    """
    Dispatch the appropriate post-processing routine to extract history output from an Abaqus `.odb` file,
    depending on the current simulation problem type.

    This function:
    - Identifies the current problem domain using `get_problem_name()`.
    - Routes the request to the corresponding routine for field-based output extraction.
    - Supports specialized routines for leaflet models (single and contact) and beam models.
    - Serves as a modular interface for automated result parsing in simulation pipelines.

    Args:
        pathName (str): Path to the directory containing the `.odb` output database and associated scripts.
        odbFileName (str): Name of the Abaqus `.odb` file to be post-processed.
        cpus (int): Number of CPUs allocated for parallel execution (currently only passed to some routines).

    Returns:
        None

    Notes:
        - Requires the global function `get_problem_name()` to resolve the current simulation configuration.
        - Expected problem names: `'leaflet_single'`, `'leaflet_contact'`, `'beam'`.
        - Delegates to:
            - `get_history_output_single(...)` for single leaflet models
            - `get_history_output_contact(...)` for contact-based leaflet models
            - `get_history_output_beam(...)` for beam models
        - This function does not return data directly; output is generated by the underlying Abaqus scripts.

    Example:
        >>> get_history_output('./simulations/', 'leaflet_model.odb', cpus=4)
    """

    problem_name = get_problem_name()
    if problem_name.lower() == 'leaflet_single':
        get_history_output_single(pathName=pathName, odbFileName=odbFileName)
    elif problem_name.lower() == 'leaflet_contact':
        get_history_output_contact(pathName=pathName, odbFileName=odbFileName, cpus=cpus)
