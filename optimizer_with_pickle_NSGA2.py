import logging
import pickle
import time

from glob2 import glob
import pathlib
from omegaconf import DictConfig
import hydra

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.callback import Callback
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.optimize import minimize
from pymoo.termination.default import DefaultMultiObjectiveTermination

from utils.global_variable import *
from utils.parce_cfg import parce_cfg
from utils.problem_no_xlsx import Procedure, init_procedure
from utils.optimisation_data_utils import *
from utils.visualize import *

config_name = 'config_leaf_NSGA2_koka'

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
    #Sets up a logger to capture verbose output, writing to both a log file and the console.
    #Args: folder_path (str): The directory path to save the log file.
    #log_file_name (str, optional): The name of the log file. Defaults to 'terminal_log.txt'.
    # Returns: logging.Logger: Configured logger for capturing terminal output.  """
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
            objectives_vars = [eval_safe(expr, vars_dict) for expr in self.obj_names]
            constraints_vars = [eval_safe(expr, vars_dict) for expr in self.constr_names]

            objectives_dict = dict(zip(self.obj_names, objectives_vars))
            constraints_dict = dict(zip(self.constr_names, constraints_vars))

            result = {
                'objectives': objectives_dict,
                'constraints': constraints_dict
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
                'HELI': objective_values['HELI'],
                'K_max': objective_values['K_max']
            }
            objectives_vars = [eval_safe(expr, vars_dict) for expr in self.obj_names]
            constraints_vars = [eval_safe(expr, vars_dict) for expr in self.constr_names]

            objectives_dict = dict(zip(self.obj_names, objectives_vars))
            constraints_dict = dict(zip(self.constr_names, constraints_vars))

            result = {
                'objectives': objectives_dict,
                'constraints': constraints_dict
            }
        elif problem_name == 'test':
            result = Procedure.run_procedure(self=self.problem, params=parameters)
            objective_values = result['objectives']
            objectives_dict = {
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
            result = {
                'objectives': objectives_dict,
                'constraints': constraints_dict
            }

        out["F"] = np.array([result['objectives'][name] for name in self.obj_names])
        out["G"] = np.array([result['constraints'][name] for name in self.constr_names])


def load_state(filepath='./', filename='checkpoint.pkl'):
    # return joblib.load(os.path.join(filepath,filename))
    with open(os.path.join(filepath, filename), "rb") as f:
        return pickle.load(f)


class CustomCallback(Callback):
    def __init__(self, objectives=['a','b'], folder_path='./utils/logs/', interval_backup=10, interval_picture=2):
        print('Callback inited')
        super().__init__()
        self.objectives = objectives
        self.folder_path = folder_path
        self.interval_backup = interval_backup
        self.interval_picture = interval_picture
        self.min_values = []

    def save_state(self, algorithm):
        with open(os.path.join(self.folder_path,f'checkpoint_{algorithm.n_gen:03d}.pkl'), 'wb') as outp:
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
                axes[idx].set_title(f"Convergence of ({obj_col}) | Δ={abs(y_values[-2]-y_values[-1]):2.2e}")
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
def main(cfg:DictConfig) -> None:
    basic_stdout = sys.stdout
    basic_stderr = sys.stderr

    parameters, objectives, constraints = parce_cfg(cfg=cfg, globalPath=str(pathlib.Path(__file__).parent.resolve()))

    print("\nRead parameters :")
    print(parameters)
    print("\nRead Objectives:")
    print(objectives)
    print("\nRead Constraints:")
    print(constraints)

    restore_state = False

    # folder to store results
    if not restore_state:
        basic_folder_path = create_results_folder(base_folder='results')
    else:
        basic_folder_path = os.path.join(pathlib.Path(__file__).parent.resolve(), 'results', 'jvalve')
        last_gen = 20
    print(f"folder path > {basic_folder_path}")

    for file in glob('./*.rpy*'):
        os.remove(file)

    # logging
    logger = setup_logger(basic_folder_path)

    start_time = time.time()
    logger.info("Starting optimization...")

    set_id(0)

    # problem initialization
    problem = Problem(parameters, objectives, constraints)

    # callback initialisation
    save_callback = CustomCallback(
        objectives=objectives,
        folder_path=basic_folder_path,
        interval_backup=10,
        interval_picture=1
    )

    # algorithm initialization
    algorithm_params = {
        "pop_size":  cfg.optimizer.pop_size,
        "n_offsprings":  cfg.optimizer.offsprings,
        "sampling": FloatRandomSampling(),
        "crossover": SBX(prob= cfg.optimizer.crossover_chance, eta= cfg.optimizer.crossover_eta),
        "mutation": PM(prob=cfg.optimizer.mutation_chance, eta=cfg.optimizer.mutation_eta),
        "eliminate_duplicates": True
    }
    algorithm = NSGA2(**algorithm_params)
    detailed_algo_params = print_algorithm_params(algorithm_params)

    # termination criteria
    termination_params = {
        "xtol": float(cfg.optimizer.termination_parameters.xtol),
        "cvtol": float(cfg.optimizer.termination_parameters.cvtol),
        "ftol": float(cfg.optimizer.termination_parameters.ftol),
        "period": float(cfg.optimizer.termination_parameters.period),
        "n_max_gen": float(cfg.optimizer.termination_parameters.n_max_gen),
        "n_max_evals": float(cfg.optimizer.termination_parameters.n_max_evals)
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
            get_problem_name(),
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

    cleanup_logger(logger)
    del logger
    sys.stdout = basic_stdout
    sys.stderr = basic_stderr


if __name__ == "__main__":
    main()