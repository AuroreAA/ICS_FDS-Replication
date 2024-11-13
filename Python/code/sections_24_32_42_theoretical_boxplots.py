import pandas as pd
import numpy as np
from ast import literal_eval
from utils.cov_cov4_functions_simplified import eig_cc4_simplified
from utils.cov_cov4_functions import eig_cc4
from utils.helpers import pad_to_length
from plotly.subplots import make_subplots
import plotly.graph_objects as go


# Paths
config_path = 'config/groups_means_grid.csv'
path_results_output = '../results/theoretical_boxplots/'
path_figures_output = '../figures/theoretical_boxplots/'

# Parameters of the distribution

dict_weights = {
    'k_2': [[0.5, 0.5],
            [0.4, 0.6],
            [0.3, 0.7],
            [0.2, 0.8],
            [0.1, 0.9]
            ],
    'k_3': [[0.33, 0.33, 0.34],
            [0.20, 0.30, 0.50],
            [0.10, 0.40, 0.50],
            [0.10, 0.30, 0.60],
            [0.10, 0.20, 0.70],
            [0.10, 0.10, 0.80]
            ],
    'k_5': [[0.2, 0.2, 0.2, 0.2, 0.2],
            [0.1, 0.2, 0.2, 0.2, 0.3],
            [0.1, 0.1, 0.2, 0.2, 0.4],
            [0.1, 0.1, 0.1, 0.3, 0.4]
            ],
    'k_10': [[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
             [0.05, 0.05, 0.05, 0.1, 0.1, 0.1, 0.1, 0.1, 0.15, 0.2],
             [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.15, 0.2, 0.3]
             ]
}

dict_weights_gaussian = {
    'k_2': [[0.5, 0.5],
            [0.2, 0.8],
            [0.1, 0.9]
            ],
    'k_3': [[0.33, 0.33, 0.34],
            [0.10, 0.40, 0.50],
            [0.10, 0.10, 0.80]
            ],
    'k_5': [[0.2, 0.2, 0.2, 0.2, 0.2],
            [0.1, 0.1, 0.2, 0.2, 0.4],
            [0.1, 0.1, 0.1, 0.1, 0.6]
            ]
}

dict_weights_threshold = {
    'k_2': [[0.3, 0.7],
            [0.21, 0.79],
            [0.1, 0.9]
            ],
    'k_3': [[0.33, 0.33, 0.34],
            [0.18, 0.32, 0.50],
            [0.10, 0.40, 0.50],
            [0.10, 0.10, 0.80]
            ],
    'k_5': [[0.2, 0.2, 0.2, 0.2, 0.2],
            [0.14, 0.2, 0.2, 0.2, 0.26],
            [0.1, 0.1, 0.2, 0.2, 0.4],
            [0.1, 0.1, 0.1, 0.3, 0.4]
            ],
    'k_10': [[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
             [0.08, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.12],
             [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.15, 0.2, 0.3]
             ]
}

param_k_q_eq = [(2, 1), (3, 2), (5, 4), (10, 9)]
param_k_q_lt = [(3, 1), (5, 1), (5, 2), (5, 3),
                (10, 1), (10, 3), (10, 5), (10, 7)]
param_k_q_gaussian = [(2, 1), (3, 2), (5, 2), (5, 4)]
param_k_q_threshold = param_k_q_eq

# For gaussian case only
p = 6

# Define the colors and names for the legend
colors = dict(zip(["rho_" + str(i+1) for i in range(10)],
                  ['#043c59', '#1b72a1', '#1d87bf', '#409ecf', '#52b5d1', '#65bdf0', '#71a6e3', '#81c3e6', '#94cceb',
                   '#c8e8fa']))

def str_to_np_array(s):
    """
    Converts a string representation of a list or array into a NumPy array.
    :param s: (str)  string representing a list, tuple, or other object that can be converted into a NumPy array.
    Example: "[1, 2, 3]" or "(4, 5, 6)".
    :return: (np.ndarray) NumPy array constructed from the literal evaluation of the input string.
    """
    return np.array(literal_eval(s))


def compute_eig_cc4_multiple_means(param_k_q, df_means, dict_weights, simplified):
    """
    Compute the eigenvalues of Cov^{-1}Cov_4 when the random variable follows a mixture of Gaussian or Dirac
    distributions. The computation is repeated for each weight scenarios dict_weights and each centers in df_means.
    This function allows different number of groups (k) and dimension of Fisher subspace (q) given by param_k_q.
    :param param_k_q: (list) tuples of pairs of number of groups (k) and dimension of Fisher subspace (q) to try
    :param df_means: (pd dataframe) one row is one configuration of group means for up to 10 groups
    :param dict_weights: (dict) for each number of group (key) there is a list of weight configuration to try (values)
    :param simplified: (bool) if True, use a mixture of Dirac, otherwise use a mixture of Gaussian
    :return df_results: (pd dataframe) one row gives the eigenvalues for a given setting (k, weights and means)
    """

    # Compute eigenvalues for each value of (k,q), each center configuration, and each weight scenario
    for param_idx, param in enumerate(param_k_q):
        k, q = param
        dict_key = 'k_' + str(k)
        results_list = []
        for i in range(df_means.shape[0]):
            means_reduced = [np.array(df_means.iloc[i, col]) for col in range(k-1)]
            means_reduced.append(np.array([0]))
            means_groups = np.vstack(
                [pad_to_length(arr, max(p, k-1)) for arr in means_reduced]  # add 0 to group means to obtain required shape
            )
            if simplified:
                means_groups = means_groups[:, :q]
                results_i = np.apply_along_axis(eig_cc4_simplified, 1, dict_weights[dict_key],
                                                mu_x_groups=means_groups, reduced=True, use_cov_inv_sqrt=True,
                                                detailed=False)
            else:
                # Update means group to fit q
                means_groups[:, q:] = 0
                results_i = np.apply_along_axis(eig_cc4, 1, dict_weights[dict_key],
                                            mu_x_groups=means_groups, reduced=False,
                                            detailed=False)
            results_list.extend(results_i.tolist())

        # Format output
        col_names = ["alpha_" + str(i+1) for i in range(k)] + ["rho_" + str(i+1) for i in range(len(results_list[0]) - k)]
        df_temp = pd.DataFrame(results_list, columns=col_names)
        df_temp.iloc[:, :k] *= 100
        df_temp.insert(0, 'alpha',
                       df_temp.iloc[:, :k].apply(lambda row: '-'.join(row.values.astype(int).astype(str)), axis=1))
        df_temp.drop(df_temp.columns[1:k + 1], axis=1, inplace=True)
        df_temp.insert(0, 'q', q)
        df_temp.insert(0, 'k', k)
        df_results = df_temp if param_idx == 0 else pd.concat([df_results, df_temp], ignore_index=True)

    return df_results


def plot_boxplot(param_k_q, df_results, dict_weights, colors, title):
    """
    Create a figure with 4 panels. Each panel is a boxplot representation of the eigenvalues of Cov^{-1}Cov_4 for
    a given number of groups (k) and dimension of Fisher subspace (q), the x-axis corresponds to different proportions
    scenarios. The figure is created from the outputs of compute_eig_cc4_multiple_means().
    :param param_k_q: (list) tuples of pairs of number of groups k and associated dimension q for each subplot
    :param df_results: (pd dataframe) one row gives the eigenvalues for a given setting (k, weights and means)
    :param dict_weights: (dict) for each number of group (key) there is a list of weight configuration to try (values)
    :param colors: (dict) for each variable name (key) there is an associated color (value) for the legend
    :param title: (str) title of the plot
    :return: plotly figure
    """

    assert len(param_k_q) == 4, "Must be 4 items in param_k_q"

    # Create a figure with 4 panels
    fig = make_subplots(rows=2, cols=2, vertical_spacing=0.15, horizontal_spacing=0.05,
                        subplot_titles=(f"k={param_k_q[0][0]}, p=q={param_k_q[0][1]}",
                                        f"k={param_k_q[1][0]}, p=q={param_k_q[1][1]}",
                                        f"k={param_k_q[2][0]}, p=q={param_k_q[2][1]}",
                                        f"k={param_k_q[3][0]}, p=q={param_k_q[3][1]}"))

    # Panel top left
    k, q = param_k_q[0]
    df_plot = df_results[(df_results['k'] == k) & (df_results['q'] == q)].copy()

    # Create a box plot trace for each column
    for column in df_plot.columns[3:]:
        if df_plot[column].isna().all():
            break
        else:
            box_trace = go.Box(x=df_plot['alpha'], y=df_plot[column], name=f'$$\\{column}$$', offsetgroup=column,
                               marker=dict(color=colors[column]), showlegend=False)
            fig.add_trace(box_trace, row=1, col=1)
    fig.add_shape(type="line", xref='paper', x0=-0.5, x1=len(dict_weights['k_' + str(k)]) - 0.5, yref="y", y0=1, y1=1,
                  line=dict(color="red", width=1.5, dash='dash'), opacity=0.5, row=1, col=1)

    # Panel top right
    k, q = param_k_q[1]
    df_plot = df_results[(df_results['k'] == k) & (df_results['q'] == q)].copy()
    # Create a box plot trace for each column
    for column in df_plot.columns[3:]:
        if df_plot[column].isna().all():
            break
        else:
            box_trace = go.Box(x=df_plot['alpha'], y=df_plot[column], name=f'$$\\{column}$$', offsetgroup=column,
                               marker=dict(color=colors[column]), showlegend=False)
            fig.add_trace(box_trace, row=1, col=2)
    fig.add_shape(type="line", xref='paper', x0=-0.5, x1=len(dict_weights['k_' + str(k)]) - 0.5, yref="y", y0=1, y1=1,
                  line=dict(color="red", width=1.5, dash='dash'), opacity=0.5, row=1, col=2)

    # Panel bottom left
    k, q = param_k_q[2]
    df_plot = df_results[(df_results['k'] == k) & (df_results['q'] == q)].copy()
    # Create a box plot trace for each column
    for column in df_plot.columns[3:]:
        if df_plot[column].isna().all():
            break
        else:
            box_trace = go.Box(x=df_plot['alpha'], y=df_plot[column], name=f'$$\\{column}$$', offsetgroup=column,
                               marker=dict(color=colors[column]), showlegend=False)
            fig.add_trace(box_trace, row=2, col=1)
    fig.add_shape(type="line", xref='paper', x0=-0.5, x1=len(dict_weights['k_' + str(k)]) - 0.5, yref="y", y0=1, y1=1,
                  line=dict(color="red", width=1.5, dash='dash'), opacity=0.5, row=2, col=1)

    # Panel bottom right
    k, q = param_k_q[3]
    df_plot = df_results[(df_results['k'] == k) & (df_results['q'] == q)].copy()
    # Create a box plot trace for each column
    for column in df_plot.columns[3:]:
        if df_plot[column].isna().all():
            break
        else:
            box_trace = go.Box(x=df_plot['alpha'], y=df_plot[column], name=f'$$\\{column}$$', offsetgroup=column,
                               marker=dict(color=colors[column]), showlegend=True)
            fig.add_trace(box_trace, row=2, col=2)
    fig.add_shape(type="line", xref='paper', x0=-0.5, x1=len(dict_weights['k_' + str(k)]) - 0.5, yref="y", y0=1, y1=1,
                  line=dict(color="red", width=1.5, dash='dash'), opacity=0.5, row=2, col=2)

    # Update layout
    ymin = df_results.iloc[:, 3:].min(axis=0).min() - 0.1
    ymax = df_results.iloc[:, 3:].max(axis=0).max() + 0.1
    fig.update_yaxes(range=[ymin, ymax])

    fig.update_xaxes(title_font=dict(size=14), tickfont=dict(size=14))
    fig.update_yaxes(title_font=dict(size=14), tickfont=dict(size=14))
    fig.update_layout(
        plot_bgcolor='rgba(229,236,246,0.5)',
        xaxis=dict(type='category'),
        legend=dict(font=dict(size=26), x=1.01, y=1, traceorder='normal'),
        margin=dict(l=40, r=80, t=40, b=40),
        title=dict(text=title, font=dict(size=18), x=0.5, xanchor='center'),
        boxmode='group',
        width=1200
    )

    for annotation in fig['layout']['annotations']:
        annotation['font'] = dict(size=18)

    return fig


if __name__ == "__main__":
    # Read config file
    df_means = pd.read_csv(config_path, sep=",")
    df_means = df_means.map(str_to_np_array)

    # Case 1: Gaussian distribution (Figure 1)
    df_results = compute_eig_cc4_multiple_means(param_k_q_gaussian, df_means, dict_weights_gaussian, False)
    df_results.to_csv(path_results_output + 'boxplot_gaussian.csv', index=False)

    fig = plot_boxplot(param_k_q_gaussian, df_results, dict_weights_gaussian, colors, title="")
    new_titles = ["k=2, q=1, p=6", "k=3, q=2, p=6", "k=5, q=2, p=6", "k=5, q=4, p=6"]
    for i, title in enumerate(new_titles):
        fig.layout.annotations[i].update(text=title)
    fig.write_image(path_figures_output + 'boxplot_gaussian' + ".jpg", scale=2)

    # Case 2: No noise, q = k-1 (Figure 2)
    df_results = compute_eig_cc4_multiple_means(param_k_q_eq, df_means, dict_weights, True)
    df_results.to_csv(path_results_output + 'boxplot_dirac_q_eq.csv', index=False)

    fig = plot_boxplot(param_k_q_eq, df_results, dict_weights, colors, title="")
    fig.update_xaxes(tickvals=[0, 1, 2],
                     ticktext=["10-10-10-10-10<br>10-10-10-10-10", "5-5-5-10-10-10<br>10-10-15-20",
                               "5-5-5-5-5-5<br>5-15-20-30"], row=2, col=2)
    fig.write_image(path_figures_output + 'boxplot_dirac_q_eq' + ".jpg", scale=2)

    # Case 3: No noise, q < k-1 (Figures 3 and S1)
    df_results = compute_eig_cc4_multiple_means(param_k_q_lt[:4], df_means, dict_weights, True)
    df_results.to_csv(path_results_output + 'boxplot_dirac_q_lt_1.csv', index=False)
    fig = plot_boxplot(param_k_q_lt[:4], df_results, dict_weights, colors, title="")
    fig.write_image(path_figures_output + 'boxplot_dirac_q_lt_1' + ".jpg")

    df_results = compute_eig_cc4_multiple_means(param_k_q_lt[-4:], df_means, dict_weights, True)
    df_results.to_csv(path_results_output + 'boxplot_dirac_q_lt_2.csv', index=False)
    fig = plot_boxplot(param_k_q_lt[-4:], df_results, dict_weights, colors, title="")
    for (l, c) in [(1, 1), (1, 2), (2, 1), (2, 2)]:
        fig.update_xaxes(tickvals=[0, 1, 2],
                         ticktext=["10-10-10-10-10<br>10-10-10-10-10", "5-5-5-10-10-10<br>10-10-15-20",
                                   "5-5-5-5-5-5<br>5-15-20-30"], row=l, col=c)
    fig.write_image(path_figures_output + 'boxplot_dirac_q_lt_2' + ".jpg", scale=2)

    # Case 4: No noise, q = k-1, at threshold (Figure 7)
    df_results = compute_eig_cc4_multiple_means(param_k_q_threshold, df_means.head(1), dict_weights_threshold, True)
    df_results.to_csv(path_results_output + 'boxplot_threshold.csv', index=False)

    fig = plot_boxplot(param_k_q_threshold, df_results, dict_weights_threshold, colors, title="")
    fig.update_xaxes(tickvals=[0, 1, 2],
                     ticktext=["10-10-10-10-10<br>10-10-10-10-10", "8-10-10-10-10<br>10-10-10-10-12",
                               "5-5-5-5-5-5<br>5-15-20-30"], row=2, col=2)
    fig.write_image(path_figures_output + 'boxplot_threshold' + ".jpg", scale=2)
