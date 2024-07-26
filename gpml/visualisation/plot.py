"""
This module provides functions to plot graphs.

Contributors:
    Julien MICHEL

Project started on:
    11/10/2022
"""
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px


def plot_importance(clf, file_name_plot, metric='all'):
    """
    Generate bar plots for feature importances based on the specified metric.

    This function generates bar plots for the feature importances based on the specified
    metric ('gain', 'weight', 'cover', or 'all'). It supports plotting multiple types of
    importances if 'all' is specified. Each plot is displayed and saved as a PNG file.

    Parameters:
    - clf (classifier): The classifier from which to get the feature importances. The
      classifier should have the `feature_names_in_` attribute and methods like
      `get_booster().get_score()`.
    - file_name_plot (str): The base name for the output plot files. This string is used
      to generate filenames for each plot.
    - metric (str, optional): The type of importance to plot. Valid options are 'gain',
      'weight', 'cover', or 'all'. Default is 'all'.

    Outputs:
    - PNG files: For each specified type of importance, a PNG file is saved in the 'fig/'
      directory. The filenames are constructed using the base name provided and the type
      of metric.
    """
    importance = pd.DataFrame()
    importance['Features'] = clf.feature_names_in_

    if metric in ('all', 'gain') :
        importance['Gain'] = clf.feature_importances_
        importance = importance.sort_values(by=['Gain'], ascending=False)
        fig = px.bar(importance, x='Gain', y='Features', color='Features', orientation='h')
        fig.update_yaxes(categoryorder="total ascending")

        fig.show()
        fig.write_image("fig/importance_gain_" + file_name_plot + ".png")

    if metric in ('all', 'weight') :
        weights = clf.get_booster().get_score(importance_type='weight')
        w = []
        for elm in importance['Features']:
            try:
                w.append(weights[elm])
            except Exception:
                w.append(0)
        importance['Weight'] = w
        importance = importance.sort_values(by=['Weight'], ascending=False)
        fig = px.bar(importance, x='Weight', y='Features', color='Features', orientation='h')
        fig.update_yaxes(categoryorder="total ascending")

        fig.show()
        fig.write_image("fig/importance_weight_" + file_name_plot + ".png")

    if metric ('all', 'cover') :
        coverages = clf.get_booster().get_score(importance_type='cover')
        c = []
        for elm in importance['Features']:
            try:
                c.append(coverages[elm])
            except Exception:
                c.append(0)
        importance['Coverage'] = c
        importance = importance.sort_values(by=['Coverage'], ascending=False)
        fig = px.bar(importance, x='Coverage', y='Features', color='Features', orientation='h')
        fig.update_yaxes(categoryorder="total ascending")

        fig.show()
        fig.write_image("fig/importance_coverage_" + file_name_plot + ".png")


def plot_features(plot_data, x_feature, y_feature, z_feature, target_label, normal_cat, file_name_plot):
    """
    Create and save a 3D scatter plot of features from the provided dataset.

    This function visualizes the relationship between three features (x, y, z) in a 3D space
    with different colors indicating different categories as specified by the 'target_label'.
    Points representing the 'normal_cat' category are colored blue, and other categories are
    colored cyclically from a preset list. The plot is saved as a PNG file.

    Parameters:
    - plot_data (DataFrame): The data containing the features to be plotted.
    - x_feature (str): The name of the column in `plot_data` to be used as the x-axis values.
    - y_feature (str): The name of the column in `plot_data` to be used as the y-axis values.
    - z_feature (str): The name of the column in `plot_data` to be used as the z-axis values.
    - target_label (str): The name of the column in `plot_data` that contains the categorical
      data used to differentiate the data points in the plot.
    - normal_cat (str): The category within `target_label` that is considered 'normal' and
      is specially colored blue.
    - file_name_plot (str): The base name for the output plot file. The plot is saved with
      this name prefixed by "high_gain_features_".

    Outputs:
    - A PNG file named "high_gain_features_<file_name_plot>.png" saved in the current directory.

    Example Usage:
    plot_features(df, 'Feature1', 'Feature2', 'Feature3', 'Category', 'Normal', 'output_plot')
    """
    fig = plt.figure(figsize=(10, 12))
    ax = fig.add_subplot(111, projection="3d")

    ax.set_xlabel(x_feature, size=16)
    ax.set_ylabel(y_feature, size=16)
    ax.set_zlabel(z_feature, size=16)

    ax.set_title("attack classes")

    c = ['b', 'g', 'c', 'm', 'y', 'k', 'r']
    c_index = 0

    for target in plot_data[target_label].unique():

        current = plot_data.loc[plot_data[target_label] == target]
        x_data = current[x_feature]
        y_data = current[y_feature]
        z_data = current[z_feature]

        if target == normal_cat:
            color = 'b'
        else:
            color = c[c_index % len(c)]

        print(target + ":" + color)
        ax.scatter(x_data, y_data, z_data, c=color)
        c_index = c_index + 1

    plt.savefig("high_gain_features_" + file_name_plot + ".png")
    plt.show()


def plot_high_gain_features(clf, file_name_plot, df_numerical_data, attack_label, targets_labels):
    """
    Identify and visualize the top three features with the highest importance from a classifier.

    This function extracts the three features with the highest importance scores from the
    provided classifier, then creates a DataFrame containing these features along with the
    attack labels. It then calls `plot_features` to generate and save a 3D visualization.

    Parameters:
    - clf (classifier): The trained model from which to get the feature importances.
    - file_name_plot (str): The base name for the output plot file.
    - df_numerical_data (DataFrame): DataFrame containing the numerical features used by
      the classifier.
    - attack_label (str): The name of the column to be used as the label for coloring the
      data points in the plot.
    - targets_labels (Series or array-like): Labels corresponding to each row in
      `df_numerical_data` to be used for coloring in the plot.

    Outputs:
    - A 3D scatter plot saved as a PNG file, which visualizes the three features with
      the highest importance scores and differentiates points by attack categories.
    """
    x, y, z = clf.feature_importances_.argsort()[-3:][::-1]
    x_feature = df_numerical_data.columns.tolist()[x]
    y_feature = df_numerical_data.columns.tolist()[y]
    z_feature = df_numerical_data.columns.tolist()[z]

    plot_data = pd.DataFrame()
    plot_data.insert(0, x_feature, df_numerical_data[x_feature], True)
    plot_data.insert(1, y_feature, df_numerical_data[y_feature], True)
    plot_data.insert(2, z_feature, df_numerical_data[z_feature], True)
    plot_data.insert(3, attack_label, targets_labels, True)  # add target

    plot_features(plot_data, x_feature, y_feature, z_feature, attack_label, 'Normal', file_name_plot)
