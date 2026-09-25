import os
import datetime
import time
import subprocess
import signal
import psutil as psu
from utils.global_variable import get_problem_name, get_cpus, get_global_path


def run_abaqus_old(
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
    subprocess.run(
        inputFile, shell=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
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
        os.chdir(get_global_path())
    return message

def _kill_process_tree(pid):
    """Kill process and all its descendants by PID."""
    try:
        parent = psu.Process(pid)
    except psu.NoSuchProcess:
        return

    try:
        children = parent.children(recursive=True)
    except psu.NoSuchProcess:
        children = []

    # Сначала дети
    for child in reversed(children):
        try:
            child.kill()
        except (psu.NoSuchProcess, psu.AccessDenied):
            pass

    # Затем родитель
    try:
        parent.kill()
    except (psu.NoSuchProcess, psu.AccessDenied):
        pass


def _get_job_processes(jobName):
    """
    Return Abaqus processes whose command line references this job.

    Processes are returned as psutil.Process objects.

    No pkill is used.
    """

    me = os.getpid()
    result = []

    jobName = str(jobName)

    for proc in psu.process_iter(
        ['pid', 'cmdline', 'name']
    ):

        try:

            info = proc.info

            if info['pid'] == me:
                continue

            cmdline = info.get('cmdline')

            if not cmdline:
                continue

            joined = ' '.join(cmdline)

            # Process belongs to this Abaqus job
            if (
                jobName in joined
                and 'abaqus' in joined.lower()
            ):
                result.append(proc)

        except (
            psu.NoSuchProcess,
            psu.AccessDenied
        ):
            continue

    return result


def _get_job_stage_pid(jobName, stage):
    """
    Find PID of a particular Abaqus stage.

    stage:
        'pre'
        'package'
        'explicit'
    """

    stage = stage.lower()

    for proc in _get_job_processes(jobName):

        try:

            name = proc.name().lower()

            if stage in name:
                return proc.pid

        except (
            psu.NoSuchProcess,
            psu.AccessDenied
        ):
            continue

    return None


def _kill_job_pids(jobName, suppress_print):
    """
    Kill only processes belonging to this Abaqus job.

    Processes are identified by PID after psutil enumeration.
    No pkill/pattern-based killing is used.
    """

    for proc in _get_job_processes(jobName):

        try:
            pid = proc.pid
            if not suppress_print:
                print(
                    f"[{jobName}] Killing PID={pid}, "
                    f"name={proc.name()}"
                )

            _kill_process_tree(pid)

        except (
            psu.NoSuchProcess,
            psu.AccessDenied
        ):
            continue


def run_abaqus(
    Path: str = None,
    jobName: str = None,
    InpFile: str = None,
    cpus: int = None,
    timeout_s=None,
    stage_timeout_s=60,
    debug=False,
    suppress_print = True,
) -> str:
    """
    Run Abaqus and monitor the Explicit solver.

    Logic:

        Abaqus launcher
              |
              +-- pre
              |
              +-- package
              |
              +-- explicit
                    |
                    +-- running
                    |
                    +-- finished

    .lck is NOT used.

    The Abaqus launcher process itself is NOT used as an indicator
    of calculation completion because it can terminate before
    `explicit` starts.

    Parameters
    ----------
    Path : str
        Abaqus working directory.

    jobName : str
        Abaqus job name.

    InpFile : str
        Input file.

    cpus : int
        Number of CPUs.

    timeout_s : float or None
        Total calculation timeout.

    stage_timeout_s : float
        Maximum allowed time for pre/package before explicit starts.

    debug : bool
        If True, Abaqus stdout/stderr are not suppressed.
    """

    if not Path:
        raise ValueError("Path is not specified")

    if not jobName:
        raise ValueError("jobName is not specified")

    if not InpFile:
        raise ValueError("InpFile is not specified")

    if not cpus:
        raise ValueError("cpus is not specified")

    # ============================================================
    # Abaqus executable
    # ============================================================

    # ============================================================
    # Command
    # ============================================================

    cmd = [
        'abaqus',
        'job=' + str(jobName),
        'inp=' + str(InpFile),
        'cpus=' + str(cpus),
        'mp_mode=threads',
        'ask_delete=OFF',
        'interactive'
    ]

    # ============================================================
    # Output
    # ============================================================

    stdout_target = (
        None
        if debug
        else subprocess.DEVNULL
    )

    # ============================================================
    # Start Abaqus
    # ============================================================

    t0 = time.monotonic()

    try:

        process = subprocess.Popen(
            cmd,
            cwd=Path,
            shell=False,
            text=True,
            stdout=stdout_target,
            stderr=stdout_target,
        )

    except Exception as e:

        raise RuntimeError(
            f"Failed to start Abaqus '{jobName}': {e}"
        ) from e
    if not suppress_print:
        print(
            f"[{jobName}] "
            f"Abaqus launcher PID={process.pid}"
        )

    # ============================================================
    # Monitoring state
    # ============================================================

    explicit_started = False

    # Don't report the same stage timeout twice
    stage_checked = False

    # ============================================================
    # Main loop
    # ============================================================

    while True:

        elapsed = time.monotonic() - t0

        # --------------------------------------------------------
        # Current stage PIDs
        # --------------------------------------------------------

        pre_pid = _get_job_stage_pid(
            jobName,
            'pre'
        )

        package_pid = _get_job_stage_pid(
            jobName,
            'package'
        )

        explicit_pid = _get_job_stage_pid(
            jobName,
            'explicit'
        )
        if not suppress_print:
            print(
                f"[{jobName}] "
                f"time={elapsed:.1f}s | "
                f"pre={pre_pid} | "
                f"package={package_pid} | "
                f"explicit={explicit_pid}"
            )

        # --------------------------------------------------------
        # Explicit appeared
        # --------------------------------------------------------

        if explicit_pid is not None:

            if not explicit_started:

                explicit_started = True
                if not suppress_print:
                    print(
                        f"[{jobName}] "
                        f"Explicit started: PID={explicit_pid}"
                    )

        # --------------------------------------------------------
        # pre/package timeout
        # --------------------------------------------------------

        if (
            not explicit_started
            and not stage_checked
            and elapsed >= stage_timeout_s
        ):

            stage_checked = True

            if pre_pid is not None:
                if not suppress_print:
                    print(
                            f"[{jobName}] "
                            f"ERROR: pre is still running "
                            f"after {stage_timeout_s}s"
                        )

                _kill_job_pids(jobName, suppress_print)

                return (
                    "ABAQUS terminated with "
                    "error in pre"
                )

            if package_pid is not None:
                if not suppress_print:
                    print(
                        f"[{jobName}] "
                        f"ERROR: package is still running "
                        f"after {stage_timeout_s}s"
                    )

                _kill_job_pids(jobName, suppress_print)

                return (
                    "ABAQUS terminated with "
                    "error in package"
                )
            if not suppress_print:
                print(
                    f"[{jobName}] "
                    f"pre/package not found; "
                    f"waiting for explicit"
                )

        # --------------------------------------------------------
        # Explicit has finished
        # --------------------------------------------------------

        if explicit_started and explicit_pid is None:

            # Give Abaqus a short moment in case the process
            # is being replaced/restarted.
            time.sleep(2)

            explicit_pid = _get_job_stage_pid(
                jobName,
                'explicit'
            )

            if explicit_pid is not None:
                if not suppress_print:
                    print(
                        f"[{jobName}] "
                        f"Explicit appeared again: "
                        f"PID={explicit_pid}"
                    )

            else:
                if not suppress_print:
                    print(
                        f"[{jobName}] "
                        f"Explicit finished"
                    )

                return "ABAQUS complete"

        # --------------------------------------------------------
        # Total timeout
        # --------------------------------------------------------

        if (
            timeout_s is not None
            and elapsed >= timeout_s
        ):
            if not suppress_print:
                print(
                    f"[{jobName}] "
                    f"Total timeout: "
                    f"{elapsed:.1f}s"
                )

            _kill_job_pids(jobName)

            return (
                "ABAQUS terminated due to "
                f"timeout ({elapsed:.1f} s)"
            )

        # --------------------------------------------------------
        # Wait
        # --------------------------------------------------------

        time.sleep(1)

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
    subprocess.run(
        consoleCommand, shell=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

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
    subprocess.run(
        consoleCommand, shell=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

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
