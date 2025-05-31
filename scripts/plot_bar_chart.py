import seaborn as sb
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

from pathlib import Path

SEED = 1234
SUBSET_RATIO = .1

BASE_PATH = Path('./')
SEED_PATH = BASE_PATH / '_data' / 'seeds'

def class_bar_plot(df: pd.DataFrame):
    # Calculate means for full dataset
    full_means = {
        'Clear': df['clear'].mean(),
        'Cloud': df['cloud'].mean(),
        'Thin Cloud': df['thin_cloud'].mean(),
        'Cloud Shadow': df['shadow'].mean()}

    # Calculate means for the subset
    subset_df = df[df['subset'] == True]
    subset_means = {
        'Clear': subset_df['clear'].mean(),
        'Cloud': subset_df['cloud'].mean(),
        'Thin Cloud': subset_df['thin_cloud'].mean(),
        'cloud_shadow': subset_df['shadow'].mean()}

    # Prepare data for the bar plot
    classes = list(full_means.keys())
    data = pd.DataFrame({
        'class': classes * 2,
        'percentage': list(full_means.values()) + list(subset_means.values()),
        'type': ['Full'] * len(classes) + ['Subset'] * len(classes)
    })

    # Plot the data
    plt.figure(figsize=(14, 8))
    sb.barplot(data=data, x='class', y='percentage', hue='type', dodge=True)
    plt.xticks(rotation=45)
    plt.title('Class Distribution in Dataset')
    plt.xlabel('Class')
    plt.ylabel('Percentage')
    
    # Move the legend to the lower right corner
    plt.legend(title="Type", loc="lower right", bbox_to_anchor=(1.13, 0))
    plt.tight_layout()
    plt.savefig(BASE_PATH / 'plots' / 'class.png', dpi=600)
    plt.show()

def biome_bar_plot(df: pd.DataFrame):
    biomes = df['biome']
    
    full_values, full_freqs = np.unique(biomes.to_numpy(), return_counts=True)
    full_freqs = full_freqs / sum(full_freqs)
    
    subset_biomes = df[df['subset'] == True]['biome']
    subset_values, subset_freqs = np.unique(subset_biomes.to_numpy(), return_counts=True)
    subset_freqs = subset_freqs / sum(subset_freqs)
    
    print('Full biomes: ', full_values)
    print('Subset biomes: ', subset_values)

    all_biomes = np.union1d(full_values, subset_values)
    full_dict = dict(zip(full_values, full_freqs))
    subset_dict = dict(zip(subset_values, subset_freqs))
    
    # Prepare data for side-by-side bars
    data = pd.DataFrame({
        'biome': np.tile(all_biomes, 2),
        'percentage': [full_dict.get(b, 0) for b in all_biomes] + [subset_dict.get(b, 0) for b in all_biomes],
        'type': ['Full'] * len(all_biomes) + ['Subset'] * len(all_biomes)
    })
    
    # Plot the data
    plt.figure(figsize=(14, 8))
    sb.barplot(data=data, x='biome', y='percentage', hue='type', dodge=True)
    plt.xticks(rotation=45)
    plt.title('Biome Distribution in Dataset')
    plt.xlabel('Biome')
    plt.ylabel('Percentage')
    
    # Move the legend to the upper right corner
    plt.legend(title="Type", loc="lower right", bbox_to_anchor=(1.13, 0))
    plt.tight_layout()
    plt.savefig(BASE_PATH / 'plots' / 'biome.png', dpi=600)

if __name__ == '__main__':
    sb.set_context("talk", rc={"axes.titlesize": 20, 
                               "axes.labelsize": 16, 
                               "legend.title_fontsize": 14, 
                               "legend.fontsize": 12})

    df = pd.read_csv(SEED_PATH / f'{SEED}_{SUBSET_RATIO}_split.csv', index_col=0)
    
    biome_bar_plot(df)
    class_bar_plot(df)