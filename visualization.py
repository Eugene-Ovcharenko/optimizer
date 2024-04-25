import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
from pymoo.indicators.hv import Hypervolume
from pymoo.problems import get_problem
from pymoo.visualization.scatter import Scatter

problem = get_problem("welded_beam")


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


def plot_convergence_by_hypervolume(
        history_df: pd.DataFrame,
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
    approx_ideal = history_df.filter(like='objective').min(axis=0)
    approx_nadir = history_df.filter(like='objective').max(axis=0)

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
        objective_columns = current_generation_df.filter(like='objective')
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


def plot_objective_convergence(
        history_df: pd.DataFrame,
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
    objective_columns = history_df.filter(like='objective').columns
    num_objectives = len(objective_columns)
    fig, axes = plt.subplots(1, num_objectives, figsize=(15, 5), sharey=False)
    if num_objectives == 1:
        axes = [axes]
    for idx, obj_col in enumerate(objective_columns):
        min_per_generation = history_df.groupby('generation')[obj_col].min()
        sns.lineplot(x=unique_generations, y=min_per_generation, ax=axes[idx],
                     marker='o', color='b', markeredgecolor=None, label=f"Best {obj_col}")
        axes[idx].set_title(f"Convergence of {obj_col}")
        axes[idx].set_xlabel("Generation")
    plt.tight_layout()
    plt.savefig(os.path.join(folder_path, 'objective_convergence.png'))


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
                s=20,
                color='b',
                alpha=0.6,
                hue=F[obj]
            )

    plt.tight_layout()
    plt.savefig(os.path.join(folder_path, 'objectives_vs_parameters.png'))


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

    fig, axes = plt.subplots(num_params, num_constrains, figsize=(30, 20), sharex=False, sharey=False)

    if num_params == 1 or num_constrains == 1:
        axes = np.array(axes).reshape((num_params, num_constrains))

    for param_idx, param in enumerate(X.columns):
        for obj_idx, constr in enumerate(G.columns):
            ax = axes[param_idx, obj_idx]
            sns.scatterplot(
                x=X[param],
                y=G[constr],
                ax=ax,
                s=20,
                color='b',
                alpha=0.6,
                hue=G[constr]
            )

    plt.tight_layout()
    plt.savefig(os.path.join(folder_path, 'constrain_vs_parameters.png'))


def plot_parallel_coordinates(
        X: pd.DataFrame,
        G: pd.DataFrame,
        F: pd.DataFrame,
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
    score = combined_data['objective1'] * combined_data['objective2']

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
