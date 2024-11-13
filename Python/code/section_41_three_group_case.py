import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils.cov_cov4_functions_simplified import eig_cc4_simplified
from utils.helpers import pad_to_length


path_results_output = '../results/three_group_case/'
path_figures_output = '../figures/three_group_case/'

k = 3
p = 2
mu_x_groups_reduced = [np.array([200]), np.array([400, 100]),  np.array([0])]


def export_results_csv(results, mu_x_groups, path_data_output, k, p, add_mu=False):
    """
    Export k_groups_results.csv and k_groups_config.csv
    :param results: (ndarray) simulation results containing groups proportions and eigenvalues
    :param mu_x_groups: (ndarray) means of the groups
    :param path_data_output: path to export csv
    :param k: number of groups
    :param p: data dimension
    """
    # Export results
    col_names_eps = ["eps_" + str(i) for i in range(k)]
    if add_mu:
        col_names = col_names_eps + ["lambda_" + str(i) for i in range(results.shape[1] - k)] + ["mu_" + str(i) for i in range(3)]
    else:
        col_names = col_names_eps + ["lambda_" + str(i) for i in range(results.shape[1] - k)]
    df_results = pd.DataFrame(results, columns=col_names)
    df_results.to_csv(path_data_output + str(k) + '_groups_results.csv', index=False)

    # Export simulation settings
    df_config = pd.DataFrame.from_dict(
        dict(zip(["mu_x" + str(i) for i in range(k)], zip(*mu_x_groups.T)))
    )
    df_config.to_csv(path_data_output + str(k) + '_groups_config.csv', index=False)

    return df_results


def create_epsilon_grid_all_permutation(k, step=0.05):
    """
    For a given number of groups k, create a grid with group proportions (epsilons) in rows: sum of each row = 1.
    All permutations are computed.
    :param k: (int) number of groups
    :param step: (real) spacing between values in the grid
    :return: (ndarray) group proportions between 0 and 1
    """

    e = np.arange(step, 1, step)
    eps_grid = np.stack(np.meshgrid(*[e for i in range(k - 1)], indexing='ij'), axis=-1).reshape(-1, k - 1)
    eps_grid = np.c_[1 - np.sum(eps_grid, axis=1), eps_grid]
    eps_grid = np.around(eps_grid, 3)
    eps_grid = eps_grid[eps_grid[:, 0] > 0]

    return eps_grid


def ternary_plot(df_results):
    """
    This function creates a ternary plot to illustrate the 3-group case. Each axis represent a group proportion. The
    colour of data points depends on the number of eigenvalues above or below 1.
    :param df_results: dataframe with simulation results with 3 groups
    """
    assert k == 3, "Must be 3 groups"

    # Color discrete map
    conditions = [
        (df_results["lambda_0"] > 1) & (df_results["lambda_1"] == 1),
        (df_results["lambda_0"] == 1) & (df_results["lambda_1"] < 1),
        (df_results["lambda_0"] > 1) & (df_results["lambda_1"] > 1),
        (df_results["lambda_0"] > 1) & (df_results["lambda_1"] < 1),
        (df_results["lambda_0"] < 1) & (df_results["lambda_1"] < 1),
    ]
    choices = [
        "ρ₁ > 1, ρ₂ = 1", "ρ₁ = 1, ρ₂ < 1", "ρ₁, ρ₂ > 1",
        "ρ₁ > 1, ρ₂ < 1", "ρ₁, ρ₂ < 1"
    ]
    color_map = {
        "ρ₁ > 1, ρ₂ = 1": '#FF2326', "ρ₁ = 1, ρ₂ < 1": '#1313FF',
        "ρ₁, ρ₂ > 1": '#FF2326', "ρ₁, ρ₂ < 1": '#1313FF',
        "ρ₁ > 1, ρ₂ < 1": '#C76BCB'
    }
    # red : '#FF2326', blue: '#1313FF', purple: '#C76BCB'
    df_results["Eigenvalues_different_than_1"] = np.select(conditions, choices, 'other')

    # Plot
    df_results.rename(columns={"eps_0": "alpha_1", "eps_1": "alpha_2", "eps_2": "alpha_3"}, inplace=True)
    fig = px.scatter_ternary(df_results, a="alpha_1", b="alpha_2", c="alpha_3", color="Eigenvalues_different_than_1",
                             color_discrete_map=color_map)

    # Legend
    fig.update_traces(showlegend=False)
    for trace in fig.data:
        if trace.name in ["ρ₁, ρ₂ > 1", "ρ₁, ρ₂ < 1", "ρ₁ > 1, ρ₂ < 1"]:
            fig.add_trace(go.Scatterternary(
                a=[None], b=[None], c=[None],
                mode='markers',
                marker=dict(size=12, color=trace.marker.color),
                showlegend=True,
                name=trace.name
            ))

    fig.update_layout(
        ternary=dict(sum=1,
                     aaxis=dict(title="α₁", title_font=dict(size=32, color='black', family='Latin Modern'),
                                tickfont=dict(size=20)),
                     baxis=dict(title="α₂", title_font=dict(size=32, color='black', family='Latin Modern'),
                                tickfont=dict(size=20)),
                     caxis=dict(title="α₃", title_font=dict(size=32, color='black', family='Latin Modern'),
                                tickfont=dict(size=20))),
        legend=dict(title="", font=dict(size=26, family='Latin Modern'))
    )

    width = 1000
    height = 800
    scale = 2

    # Save
    path = path_figures_output + "Fig5_discrete_ternary_plot.jpg"
    fig.write_image(path, width=width, height=height, scale=scale)


def ternary_plot_gradient(df_results, lambda_1=True):
    """
    This function creates a ternary plot to illustrate the 3-group case. Each axis represent a group proportion. The
    colour of data points depends on the value of the first (if lambda_1=True) or second (if lambda_1=False) eigenvalue
    of Cov^{-1}Cov_4. It follows a continuous color scale: red if the eigenvalue is below 1, white if it is equal to 1
    and blue if it is above 1.
    :param df_results: dataframe with simulation results with 3 groups
    :param lambda_1: (bool) if true (default) use the first eigenvalue, otherwise use the second one
    """
    assert k == 3, "Must be 3 groups"

    # Define the color scale: the log of the eigenvalue is used instead of the eigenvalue
    df_results['log_lambda_0'] = np.log(df_results['lambda_0'])
    df_results['log_lambda_1'] = np.log(df_results['lambda_1'])
    if lambda_1:
        df_results['log_lambda'] = df_results['log_lambda_0']
    else:
        df_results['log_lambda'] = df_results['log_lambda_1']
    min_log_lambda = df_results[['log_lambda_0', 'log_lambda_1']].min().min()
    max_log_lambda = df_results[['log_lambda_0', 'log_lambda_1']].max().max()
    df_results.rename(columns={"eps_0": "alpha_1", "eps_1": "alpha_2", "eps_2": "alpha_3"}, inplace=True)

    mid_relative_position = (0 - min_log_lambda) / (max_log_lambda - min_log_lambda)
    color_scale = [
        [0, 'blue'],  # min value
        [mid_relative_position, 'white'],  # midpoint (log_lambda=0)
        [1, 'red']  # max value
    ]

    # Legend parameters: even if the log is used to create the scale, the eigenvalue is displayed in the legend
    lambda_ticks = [-1, 0, 1, 2, 3, 4, 5]
    lambda_ticklabels = [np.round(np.exp(val), 2) if i == 0 else np.round(np.exp(val))
                         for i, val in enumerate(lambda_ticks)]

    # Create plot
    fig = px.scatter_ternary(df_results, a="alpha_1", b="alpha_2", c="alpha_3", color='log_lambda',
                             color_continuous_scale=color_scale,
                             range_color=[min_log_lambda, max_log_lambda])
    fig.update_coloraxes(colorbar_title="ρ&nbsp;&nbsp;", colorbar_ticks='outside', colorbar_tickvals=lambda_ticks,
                         colorbar_ticktext=lambda_ticklabels)
    fig.update_layout(
        ternary=dict(sum=1,
                     aaxis=dict(title="α₁", title_font=dict(size=38, color='black', family='Latin Modern'), tickfont=dict(size=26)),
                     baxis=dict(title="α₂", title_font=dict(size=38, color='black', family='Latin Modern'), tickfont=dict(size=26)),
                     caxis=dict(title="α₃", title_font=dict(size=38, color='black', family='Latin Modern'), tickfont=dict(size=26))),
        coloraxis_colorbar=dict(title_font=dict(size=34), tickfont=dict(size=26),
                                orientation='h', yanchor='bottom', y=-0.3, x=0.5, xanchor='center')
    )

    width = 1000
    height = 800
    scale = 2

    # Save
    if lambda_1:
        path = path_figures_output + "Fig4a_ternary_plot_gradient_rho1" + ".jpg"
    else:
        path = path_figures_output + "Fig4b_ternary_plot_gradient_rho2" + ".jpg"
    fig.write_image(path, width=width, height=height, scale=scale)


if __name__ == "__main__":

    # Simulation over different group proportion scenarios

    eps_grid = create_epsilon_grid_all_permutation(k, step=0.001)
    mu_x_groups_reduced = [mu * 1 for mu in mu_x_groups_reduced]
    mu_x_groups = np.vstack(
        [pad_to_length(arr, p) for arr in mu_x_groups_reduced]  # add 0 to group means to obtain required shape
    )

    results = np.apply_along_axis(eig_cc4_simplified, 1, eps_grid, mu_x_groups)
    df_results = export_results_csv(results, mu_x_groups, path_results_output, k, p)

    # Figure 4a
    ternary_plot_gradient(df_results, lambda_1=True)
    # Figure 4b
    ternary_plot_gradient(df_results, lambda_1=False)
    # Figure 5
    ternary_plot(df_results)

