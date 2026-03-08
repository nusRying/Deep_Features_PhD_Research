import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def moving_average(a, threshold=300):
    """
    Computes the moving average of an array.
    Used for smoothing match set and correct set sizes.
    """
    if len(a) < threshold:
        return a
    weights = np.repeat(1.0, threshold) / threshold
    conv = np.convolve(a, weights, 'valid')
    # Append the last valid value to maintain original length
    return np.append(conv, np.full(threshold - 1, conv[conv.size - 1]))

def cumulative_frequency(freq):
    """
    Computes the cumulative frequency of an operation count.
    Used for visualizing GA and covering operations over time.
    """
    a = []
    c = []
    for i in freq:
        a.append(i + sum(c))
        c.append(i)
    return np.array(a)

def plot_performance(df, save_path=None):
    """Plots Accuracy and Generality vs Iterations."""
    plt.figure(figsize=(10, 6))
    plt.plot(df['Iteration'], df['Accuracy (approx)'], label="Approx. Accuracy")
    plt.plot(df['Iteration'], df['Average Population Generality'], label="Avg. Generality")
    plt.xlabel('Iteration')
    plt.ylabel('Score')
    plt.title('Training Performance Dynamics')
    plt.legend()
    if save_path:
        plt.savefig(os.path.join(save_path, 'performance.png'))
    plt.close()

def plot_population(df, save_path=None):
    """Plots Macropopulation and Micropopulation sizes vs Iterations."""
    plt.figure(figsize=(10, 6))
    plt.plot(df['Iteration'], df['Macropopulation Size'], label="Macropopulation Size")
    plt.plot(df['Iteration'], df['Micropopulation Size'], label="Micropopulation Size")
    plt.xlabel('Iteration')
    plt.ylabel('Population Size')
    plt.title('Population Dynamics')
    plt.legend()
    if save_path:
        plt.savefig(os.path.join(save_path, 'population.png'))
    plt.close()

def plot_sets(df, threshold=300, save_path=None):
    """Plots Match Set [M] and Correct Set [C] sizes with moving averages."""
    plt.figure(figsize=(10, 6))
    m_size = df['Match Set Size'].values
    c_size = df['Correct Set Size'].values
    
    plt.plot(df['Iteration'], m_size, alpha=0.3, label="[M] Size", color='blue')
    plt.plot(df['Iteration'], moving_average(m_size, threshold), label=f"[M] Size (Avg {threshold})", color='blue', linewidth=2)
    
    plt.plot(df['Iteration'], c_size, alpha=0.3, label="[C] Size", color='green')
    plt.plot(df['Iteration'], moving_average(c_size, threshold), label=f"[C] Size (Avg {threshold})", color='green', linewidth=2)
    
    plt.xlabel('Iteration')
    plt.ylabel('Set Size')
    plt.title('Match and Correct Set Sizes')
    plt.legend()
    if save_path:
        plt.savefig(os.path.join(save_path, 'set_sizes.png'))
    plt.close()

def plot_operations(df, save_path=None):
    """Plots cumulative count of various LCS operations."""
    plt.figure(figsize=(12, 8))
    
    ops = {
        '# Classifiers Subsumed in Iteration': 'Subsumption',
        '# Crossover Operations Performed in Iteration': 'Crossover',
        '# Mutation Operations Performed in Iteration': 'Mutation',
        '# Covering Operations Performed in Iteration': 'Covering',
        '# Deletion Operations Performed in Iteration': 'Deletion',
        '# Rules Removed via Rule Compaction': 'Rule Compaction'
    }
    
    for col, label in ops.items():
        if col in df.columns:
            plt.plot(df['Iteration'], cumulative_frequency(df[col].values), label=label)
            
    plt.xlabel('Iteration')
    plt.ylabel('Cumulative Operation Count')
    plt.title('LCS Operations Analytics')
    plt.legend()
    if save_path:
        plt.savefig(os.path.join(save_path, 'operations.png'))
    plt.close()

def plot_timing(df, save_path=None):
    """Plots stacked timing analysis of different training phases."""
    plt.figure(figsize=(12, 8))
    
    iterations = df['Iteration'].values
    
    # Timing components as defined in the notebook stack
    components = [
        ('Total Model Initialization Time', 'Init'),
        ('Total Matching Time', 'Matching'),
        ('Total Covering Time', 'Covering'),
        ('Total Selection Time', 'Selection'),
        ('Total Crossover Time', 'Crossover'),
        ('Total Mutation Time', 'Mutation'),
        ('Total Subsumption Time', 'Subsumption'),
        ('Total Attribute Tracking Time', 'Attribute Tracking'),
        ('Total Deletion Time', 'Deletion'),
        ('Total Rule Compaction Time', 'Rule Compaction'),
        ('Total Evaluation Time', 'Evaluation')
    ]
    
    current_stack = np.zeros(len(df))
    
    for col, label in components:
        if col in df.columns:
            values = df[col].values
            plt.plot(iterations, current_stack + values, label=f"{label} Time")
            current_stack += values
            
    # Add the total time line
    if 'Total Global Time' in df.columns:
        plt.plot(iterations, df['Total Global Time'].values, 'k--', label="Total Global Time", alpha=0.7)
        
    plt.xlabel('Iteration')
    plt.ylabel('Cumulative Time (Seconds)')
    plt.title('Training Phase Timing Analysis (Stacked)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(os.path.join(save_path, 'timing.png'))
    plt.close()
