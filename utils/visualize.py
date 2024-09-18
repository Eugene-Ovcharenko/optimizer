import os
from typing import Tuple, Optional, List, Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from pymoo.indicators.hv import Hypervolume
from pymoo.problems import get_problem
from pymoo.visualization.scatter import Scatter
from pymoo.decomposition.asf import ASF
from .global_variable import get_problem_name, get_s_lim
import matplotlib.colors as mcolors
import matplotlib as mpl

problem = get_problem("welded_beam")


def create_pareto_front_plot(
        csv_path: str,
        data: pd.DataFrame,
        folder_path: str,
        objectives: list,
        pymoo_problem: str = "welded_beam"
) -> None:
    def create_pareto_front_plot_pymoo(
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
        pymoo_scatter_plot.add(get_problem(pymoo_problem).pareto_front(use_cache=False), plot_type="line",
                               color="black")
        pymoo_scatter_plot.add(data.values, facecolor="none", edgecolor="red", alpha=0.8, s=20)
        pymoo_scatter_plot.save(os.path.join(folder_path, 'pareto_front.png'))

    def create_pareto_front_plot_fea(
            csv_path: str,
            objectivies: list,
            folder_path: str,
    ) -> None:
        """
        Creates and saves a scatter plot showing the Pareto front for a specified problem along with additional data.

        Args:
            csv_path (str): Path to the CSV file with the FEA results.
            folder_path (str): The folder path where the plot will be saved.
            pymoo_problem (str): The name of the problem to get the Pareto front from. Defaults to "welded_beam".

        Returns:
            None: The function saves the plot to the specified folder and does not return any value.
        """

        # Load FEA results from CSV
        data = pd.read_csv(csv_path)

        # Extract objective values as a list of tuples
        # try:
        #     points = list(zip(data['LMN_open'], data['LMN_closed'], data['Smax']))
        # except:
        #     try:
        #         points = list(zip(data['LMN_open']))
        #     except:
        #         try:
        #             points = list(zip(data['LMN_closed']))
        #         except:
        #             try:
        #                 points = list(zip(data['Smax']))
        #             except:
        #                 raise('cannot load objcetives')
        points = objectivies

        def is_dominated(point, others):
            return any(
                (other[0] < point[0] and other[1] <= point[1] and other[2] <= point[2]) or
                (other[0] <= point[0] and other[1] < point[1] and other[2] <= point[2]) or
                (other[0] <= point[0] and other[1] <= point[1] and other[2] < point[2])
                for other in others
            )

        non_dominated_points = [point for point in points if not is_dominated(point, points)]

        # Convert non-dominated points to a DataFrame
        pareto_df = pd.DataFrame(non_dominated_points, columns=objectivies)

        # Function to create and save a plot with a given view angle
        def save_plot_with_angle(elev, azim, filename):
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(data['LMN_open'], data['LMN_closed'], data['Smax'], label='Solutions', alpha=0.5)

            # Creating a surface plot for Pareto front
            X, Y, Z = pareto_df['LMN_open'], pareto_df['LMN_closed'], pareto_df['Smax']
            ax.plot_trisurf(X, Y, Z, color='r', alpha=0.4, linewidth=0.2)

            ax.set_xlabel('LMN_open')
            ax.set_ylabel('LMN_closed')
            ax.set_zlabel('Smax')
            ax.set_title('Pareto Front for FEA Results')
            ax.legend()

            # Rotate the plot to the specified angle
            ax.view_init(elev=elev, azim=azim)

            # Save the plot to the specified folder
            plot_path = os.path.join(folder_path, filename)
            plt.savefig(plot_path)
            plt.close()

        # Ensure the folder exists
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # Save the plot from three different orientations
        if len(objectivies) == 3:
            save_plot_with_angle(30, 90, 'pareto_front_rotated_1.png')
            save_plot_with_angle(30, 180, 'pareto_front_rotated_2.png')
            save_plot_with_angle(30, 270, 'pareto_front_rotated_3.png')

    problem_name = get_problem_name().lower()
    if problem_name == 'test':
        create_pareto_front_plot_pymoo(data=data, folder_path=folder_path, pymoo_problem=pymoo_problem)
    else:
        create_pareto_front_plot_fea(csv_path=csv_path, objectivies=objectives, folder_path=folder_path)


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
    plt.close()


def plot_convergence_by_hypervolume(
        history_df: pd.DataFrame,
        objectives: list[str],
        folder_path: str,
        ref_point: np.ndarray = np.array([1.0, 1.0]),
) -> None:
    """
    Plots convergence by hypervolume and saves it to a specified folder.

    Args:
        history_df (pd.DataFrame): The DataFrame containing optimization history.
        folder_path (str): The directory path to save the plot.
        ref_point (np.ndarray, optional): The reference point for
                   hypervolume calculation. Defaults to np.array([1.0, 1.0]).

    Returns:
        None: This function creates a plot and saves it to the specified file.
    """
    approx_ideal = history_df[objectives].min(axis=0)
    approx_nadir = history_df[objectives].max(axis=0)

    hv_metric = Hypervolume(
        ref_point=ref_point,
        ideal=approx_ideal,
        nadir=approx_nadir,
        zero_to_one=True,
        norm_ref_point=True
    )
    unique_generations = history_df['generation'].unique()
    objective_values_per_generation = []
    for gen in unique_generations:
        current_generation_df = history_df[history_df['generation'] == gen]
        objective_columns = current_generation_df[objectives]
        objective_values_per_generation.append(objective_columns.to_numpy())
    hypervolume_values = [hv_metric.do(_F) for _F in objective_values_per_generation]
    n_evals = list(range(1, len(hypervolume_values) + 1))

    sns.set(style="whitegrid")
    plt.figure(figsize=(7, 5))
    sns.lineplot(x=n_evals, y=hypervolume_values, marker='.', linestyle='-', color='b', markersize=10)
    plt.title("Convergence by Hypervolume")
    plt.xlabel("Function Evaluations")
    plt.ylabel("Hypervolume")
    plt.tight_layout()
    plt.savefig(os.path.join(folder_path, 'convergence_by_hypervolume.png'))
    plt.close()


def plot_objective_convergence(
        history_df: pd.DataFrame,
        objectives: list[str],
        folder_path: str
) -> None:
    """
    Plots convergence of objectives over generations, with optional modes to visualize  best objective values.

    Args:
        history_df (pd.DataFrame): DataFrame containing optimization history.
        folder_path (str): Directory path to save the plot.

    Returns:
        None: This function creates and saves a plot showing objective convergence over generations.
    """
    unique_generations = sorted(history_df['generation'].unique())
    objective_columns = history_df[objectives].columns
    num_objectives = len(objective_columns)
    fig, axes = plt.subplots(1, num_objectives, figsize=(15, 5), sharey=False)
    if num_objectives == 1:
        axes = [axes]
    for idx, obj_col in enumerate(objective_columns):
        if obj_col.lower() == 'lmn_open':
            obj_col_print = '1 - LMN_open'
        else:
            obj_col_print = obj_col
        min_per_generation = history_df.groupby('generation')[obj_col].min()
        sns.lineplot(x=unique_generations, y=min_per_generation, ax=axes[idx],
                     marker='o', color='b', markeredgecolor=None, label=f"Best {obj_col_print}")
        axes[idx].set_title(f"Convergence of ({obj_col_print})")
        axes[idx].set_xlabel("Generation")
    plt.tight_layout()
    plt.savefig(os.path.join(folder_path, 'objective_convergence.png'))
    plt.close()


def plot_objectives_vs_parameters(
        X: pd.DataFrame,
        F: pd.DataFrame,
        folder_path: str
) -> None:
    """
    Plots scatter plots of each objective against each parameter in subplots.

    Args:
        X (pd.DataFrame): DataFrame containing parameter values.
        F (pd.DataFrame): DataFrame containing objective values.
        folder_path (str): Directory path to save the plot.

    Returns:
        None: This function creates scatter plots and saves them to the specified file.
    """
    num_params = len(X.columns)
    num_objectives = len(F.columns)

    fig, axes = plt.subplots(num_params, num_objectives, figsize=(15, 10), sharex=False, sharey=False)

    if num_params == 1 or num_objectives == 1:
        axes = np.array(axes).reshape((num_params, num_objectives))

    for param_idx, param in enumerate(X.columns):
        for obj_idx, obj in enumerate(F.columns):
            ax = axes[param_idx, obj_idx]
            sns.scatterplot(
                x=X[param],
                y=F[obj],
                ax=ax,
                s=30,
                color='b',
                alpha=0.8,
                hue=False,
                legend=False
            )

    plt.tight_layout()
    plt.savefig(os.path.join(folder_path, 'objectives_vs_parameters.png'))
    plt.close()


def plot_constrains_vs_parameters(
        X: pd.DataFrame,
        G: pd.DataFrame,
        folder_path: str
) -> None:
    """
    Plots scatter plots of each constrain against each parameter in subplots.

    Args:
        X (pd.DataFrame): DataFrame containing parameter values.
        G (pd.DataFrame): DataFrame containing constrain values.
        folder_path (str): Directory path to save the plot.

    Returns:
        None: This function creates scatter plots and saves them to the specified file.
    """

    num_params = len(X.columns)
    num_constrains = len(G.columns)

    fig, axes = plt.subplots(num_params, num_constrains, figsize=(20, 10), sharex=False, sharey=False)

    if num_params == 1 or num_constrains == 1:
        axes = np.array(axes).reshape((num_params, num_constrains))

    for param_idx, param in enumerate(X.columns):
        for obj_idx, constr in enumerate(G.columns):
            ax = axes[param_idx, obj_idx]
            sns.scatterplot(
                x=X[param],
                y=G[constr],
                ax=ax,
                s=30,
                color='b',
                alpha=0.8,
                hue=False,
                legend=False
            )

    plt.tight_layout()
    plt.savefig(os.path.join(folder_path, 'constrain_vs_parameters.png'))
    plt.close()


def plot_parallel_coordinates(
        X: pd.DataFrame,
        G: pd.DataFrame,
        F: pd.DataFrame,
        objectives: list[str],
        folder_path: str,
        file_name: str = 'parallel_coordinates.html'
) -> None:
    """
    Creates an interactive parallel coordinates plot for parameters, constraints,
    and objectives, and saves it to a specified folder.

    Args:
        X (pd.DataFrame): DataFrame containing parameter values.
        G (pd.DataFrame): DataFrame containing constraint values.
        F (pd.DataFrame): DataFrame containing objective values.
        folder_path (str): Directory path to save the plot.
        file_name (str, optional): Name of the file for saving the plot. Defaults to 'parallel_coordinates.html'.

    Returns:
        None: This function creates and saves an interactive parallel coordinates plot.
    """

    combined_data = pd.concat([X, G, F], axis=1)
    score = combined_data[objectives[0]]
    for obj in objectives[1:-1]:
        score *= combined_data[obj]

    fig = px.parallel_coordinates(
        combined_data,
        dimensions=combined_data.columns,
        color=score,
        color_continuous_scale=px.colors.diverging.Tealrose,
        labels={col: col for col in combined_data.columns},
        title="Parallel Coordinates Plot",
        width=1024,
        height=768
    )
    plot_path = os.path.join(folder_path, file_name)
    fig.write_html(plot_path)


def plot_best_objectives(
        F: pd.DataFrame,
        folder_path: str,
        weights: list[float] or str = 'equal'
) -> None:
    """
    Finds the best trade-off in a multi-objective optimization based on ASF and creates subplots
    for each unique pair of objectives to visualize the best solutions.

    Args:
        F (pd.DataFrame): DataFrame containing the objective values.
        weights (list[float] or str): A list of weights to assign to each objective, or 'equal' for equal weights. Defaults to 'equal'.
        folder_path (str): The directory path to save the plots.

    Returns:
        None: The function plots and saves scatter plots with highlighted best solutions.
    """
    num_objectives = F.shape[1]

    if weights == 'equal':  # Determine weights based on the input
        weights = [1 / num_objectives] * num_objectives
    elif isinstance(weights, list) and len(weights) == num_objectives:
        if not np.isclose(sum(weights), 1):
            raise ValueError("List of weights must sum to 1.")
    else:
        raise ValueError("Weights must be either 'equal' or a list of correct length.")

    # Normalize the objective values for ASF
    approx_ideal = F.min(axis=0)
    approx_nadir = F.max(axis=0)
    nF = (F - approx_ideal) / (approx_nadir - approx_ideal)
    decomp = ASF()
    best_index = decomp.do(nF.to_numpy(), 1 / np.array(weights)).argmin()

    subplot_count = (num_objectives * (num_objectives - 1)) // 2  # Unique pairs of objectives
    fig, axes = plt.subplots(1, subplot_count, figsize=(5 * subplot_count, 5))
    plot_idx = 0

    for i in range(num_objectives):
        for j in range(i + 1, num_objectives):  # Only create unique pairs
            ax = axes[plot_idx] if subplot_count > 1 else axes
            sns.scatterplot(
                data=F,
                x=F.columns[i],
                y=F.columns[j],
                label="All points",
                s=30,
                ax=ax,
                color='blue',
                alpha=0.6
            )
            sns.scatterplot(
                data=F.iloc[[best_index]],
                x=F.columns[i],
                y=F.columns[j],
                label="Best point",
                s=200,
                marker="x",
                color="red",
                ax=ax
            )
            ax.set_title(f"Objective ({F.columns[i]}) vs Objective ({F.columns[j]})")
            ax.set_xlabel(F.columns[i])
            ax.set_ylabel(F.columns[j])
            plot_idx += 1

    plt.tight_layout()
    plt.savefig(os.path.join(folder_path, 'objective_space_subplots.png'))
    plt.close()


def load_optimization_results(
        folder_path: str,
        csv_files: Dict[str, str]
) -> Dict[str, Optional[pd.DataFrame]]:
    """
    Load optimization results from CSV files in a specified folder.

    Args:
        folder_path (str): The path to the folder where CSV files are stored.
        csv_files (Dict[str, str]): A dictionary mapping DataFrame names to CSV file names.

    Returns:
        Dict[str, Optional[pd.DataFrame]]: A dictionary with keys representing the names of DataFrames
            and values being the loaded DataFrames or `None` if the file is not found.
    """
    dataframes = {}
    for df_name, csv_file in csv_files.items():
        try:
            dataframes[df_name] = pd.read_csv(os.path.join(folder_path, csv_file), index_col=0)
        except FileNotFoundError:
            print(f"Warning: '{csv_file}' not found in '{folder_path}'.")
            dataframes[df_name] = None

    return dataframes


def colored(text, color):
    if color == "green":
        return f"\033[92m{text}\033[0m"  # Green text
    elif color == "red":
        return f"\033[91m{text}\033[0m"  # Red text
    return text


def plot_parameters_over_generation(
        data: pd.DataFrame,
        parameters: list[str],
        objectives: list[str],
        folder_path: str
):
    def plot_parameter_with_hue(axis, parameter, colors, title):
        figure = axis.scatter(data['generation'], data[parameter], c=colors, alpha=0.5, marker='.', edgecolors=None)
        axis.set_title(title)
        axis.set_xlabel('Generation')
        axis.set_ylabel(parameter)

        return figure

    # Create the colormap with red for lower values and blue for higher values
    cmap_red_blue = plt.cm.ScalarMappable(norm=mcolors.Normalize(vmin=0, vmax=1), cmap='coolwarm_r')

    # Normalize the objectives to create a color map
    # Convert normalized values to colors using the red to blue colormap

    n_param = len(parameters)
    n_objectives = len(objectives)

    # plt.figure(figsize=(60, 10))
    f, ax = plt.subplots(n_objectives, n_param, figsize=(10 * n_param, 10))
    plt.set_cmap('coolwarm_r')
    for iter_obj in range(n_objectives):
        norm_obj = mcolors.Normalize(vmin=data[objectives[iter_obj]].min(), vmax=data[objectives[iter_obj]].max())
        colors_obj = cmap_red_blue.to_rgba(norm_obj(data[objectives[iter_obj]]))
        if iter_obj in [1, 2]:
            ylim = 1
        else:
            ylim = get_s_lim()
        for iter_param in range(n_param):
            # Plot parameters over generations with hue based on 1 - LMN_open
            #print(f'({parameters[iter_param]}) over Generations (Hue: {objectives[iter_obj]})')
            fig = plot_parameter_with_hue(
                ax[iter_obj, iter_param],
                parameters[iter_param],
                colors_obj,
                f'({parameters[iter_param]}) over Generations (Hue: {objectives[iter_obj]})'
            )
        cbar = f.colorbar(fig, ax=ax[iter_obj, -1], location='right', cmap=cmap_red_blue)
        cbar.set_ticks([fig.colorbar.vmin + t * (fig.colorbar.vmax - fig.colorbar.vmin) for t in cbar.ax.get_yticks()])
        cbar.set_ticklabels(
            np.round(
                np.linspace(
                    np.min(data[objectives[iter_obj]]),
                    np.max(data[objectives[iter_obj]]),
                    6,
                    endpoint=True),
                3
            )
        )
    plt.tight_layout()
    plt.savefig(os.path.join(folder_path, 'parameters over generations.png'))

    # # Plot parameters over generations with hue based on Smax
    # plt.figure(figsize=(14, 10))
    #
    # plt.subplot(2, 2, 1)
    # plot_parameter_with_hue(plt.gca(), 'Lstr', colors_smax, 'Lstr over Generations (Hue: Smax)')
    #
    # plt.subplot(2, 2, 2)
    # plot_parameter_with_hue(plt.gca(), 'ANG', colors_smax, 'ANG over Generations (Hue: Smax)')
    #
    # plt.subplot(2, 2, 3)
    # plot_parameter_with_hue(plt.gca(), 'CVT', colors_smax, 'CVT over Generations (Hue: Smax)')
    #
    # plt.subplot(2, 2, 4)
    # plot_parameter_with_hue(plt.gca(), 'LAS', colors_smax, 'LAS over Generations (Hue: Smax)')
    #
    # plt.tight_layout()
    # plt.show()


from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from scipy.interpolate import interp1d
from scipy.interpolate import make_interp_spline
def plot_pareto_with_trade_off(
        history: pd.DataFrame,
        F: pd.DataFrame,
        objectives: list[str],
        folder_path: str,
        weights: list[float] or str = 'equal'
) -> None:
    """
    Finds the best trade-off in a multi-objective optimization based on ASF and creates subplots
    for each unique pair of objectives to visualize the best solutions.

    Args:
        F (pd.DataFrame): DataFrame containing the objective values.
        weights (list[float] or str): A list of weights to assign to each objective, or 'equal' for equal weights. Defaults to 'equal'.
        folder_path (str): The directory path to save the plots.

    Returns:
        None: The function plots and saves scatter plots with highlighted best solutions.
    """

    num_objectives = F.shape[1]
    history
    if weights == 'equal':  # Determine weights based on the input
        weights = [1 / num_objectives] * num_objectives
    elif isinstance(weights, list) and len(weights) == num_objectives:
        if not np.isclose(sum(weights), 1):
            raise ValueError("List of weights must sum to 1.")
    else:
        raise ValueError("Weights must be either 'equal' or a list of correct length.")

    # Normalize the objective values for ASF
    approx_ideal = F.min(axis=0)
    approx_nadir = F.max(axis=0)
    nF = (F - approx_ideal) / (approx_nadir - approx_ideal)
    decomp = ASF()
    best_index = decomp.do(nF.to_numpy(), 1 / np.array(weights)).argmin()

    # Identify the Pareto front
    pareto_indices = NonDominatedSorting().do(F.to_numpy(), only_non_dominated_front=True)
    pareto_front = F.iloc[pareto_indices]

    subplot_count = (num_objectives * (num_objectives - 1)) // 2  # Unique pairs of objectives
    fig, axes = plt.subplots(1, subplot_count, figsize=(10 * subplot_count, 10))
    plot_idx = 0

    for i in range(num_objectives):
        for j in range(i + 1, num_objectives):  # Only create unique pairs
            ax = axes[plot_idx] if subplot_count > 1 else axes
            sns.scatterplot(
                data=history[objectives],
                x=F.columns[i],
                y=F.columns[j],
                label="All points",
                s=20,
                ax=ax,
                color='blue',
                alpha=0.6
            )
            sns.scatterplot(
                data=F.iloc[[best_index]],
                x=F.columns[i],
                y=F.columns[j],
                label="Best point",
                s=200,
                marker="x",
                color="red",
                ax=ax
            )

            # Sort the Pareto front points for continuous line plotting
            pareto_sorted = pareto_front.sort_values(by=[F.columns[i], F.columns[j]])
            ax.plot(pareto_sorted[F.columns[i]], pareto_sorted[F.columns[j]], label="Pareto front", color="green")

            ax.set_title(f"Objective ({F.columns[i]}) vs Objective ({F.columns[j]})")
            ax.set_xlabel(F.columns[i])
            ax.set_ylabel(F.columns[j])
            plot_idx += 1

    plt.tight_layout()
    plt.savefig(os.path.join(folder_path, 'pareto_front.png'))
    plt.close()
