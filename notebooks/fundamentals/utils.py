import itertools
from typing import List, Optional, Tuple

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.gridspec import GridSpec

# Define a function to validate inputs manually
def validate_plot_config(pairs: List[Tuple[str, str]], data: pd.DataFrame, save_path: Optional[str] = None):
    # Ensure pairs contain valid column names in the DataFrame
    for x, y in pairs:
        if x not in data.columns or y not in data.columns:
            raise ValueError(f"Columns '{x}' and/or '{y}' not found in the DataFrame.")
    # Check if save_path is a string if provided
    if save_path is not None and not isinstance(save_path, str):
        raise TypeError("`save_path` should be a string representing a file path.")
    return pairs, data, save_path

def plot_regression_and_heatmap_plots(
    pairs: List[Tuple[str, str]], 
    data: pd.DataFrame, 
    save_path: Optional[str] = None,
    marker_size: int = 200,          # Size of the markers in regplot
    alpha: float = 0.5,              # Transparency of the markers in regplot
    annot_size: int = 40,            # Font size of annotations in heatmap
    cbar_fontsize: int = 40,         # Font size for color bar
    cbar_title: str = 'Pearson Correlation Coefficient',  # Title for color bar
    cbar_title_pad: float = 40        # Padding for the color bar title
):
    # Run manual validation
    pairs, data, save_path = validate_plot_config(pairs, data, save_path)

    # Create the plot
    fig = plt.figure(figsize=(25, 20), dpi=300)
    gs = GridSpec(3, 2, width_ratios=[3, 2]) 
    colors = sns.color_palette("husl", len(pairs))
    
    for i, ((x, y), color) in enumerate(zip(pairs, colors)):
        ax = fig.add_subplot(gs[i, 0])
        sns.regplot(x=x, y=y, data=data, ax=ax, 
                    scatter_kws={'alpha': alpha, 's': marker_size}, line_kws={'color': color})
        ax.set_title(f'Regression plot: {x} vs {y}', fontsize=40)
        ax.tick_params(axis='x', rotation=45)
        ax.xaxis.label.set_size(40)
        ax.yaxis.label.set_size(40)
        ax.tick_params(axis='x', labelsize=40, size=20)
        ax.tick_params(axis='y', labelsize=40, size=20)
    
    ax = fig.add_subplot(gs[:, 1]) 
    corr_matrix = data.corr()
    cbar = sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', cbar=True, 
                       square=True, ax=ax, annot_kws={"size": annot_size})
    ax.set_title('Correlation Heatmap', fontsize=40)
    ax.tick_params(axis='x', size=20, labelsize=40)
    ax.tick_params(axis='y', size=20, labelsize=40)
    
    # Adjust color bar font size and add title
    colorbar = cbar.collections[0].colorbar
    colorbar.ax.tick_params(labelsize=cbar_fontsize)  # Set color bar tick font size
    colorbar.set_label(cbar_title, fontsize=cbar_fontsize, labelpad=cbar_title_pad, rotation=270)  # Set color bar title and padding
    colorbar.ax.yaxis.label.set_size(cbar_fontsize)  # Ensure color bar label size is set
    
    plt.tight_layout()
    plt.show()
    
    # Save if save_path is provided
    if save_path:
        fig.savefig(save_path)
    
    return fig


