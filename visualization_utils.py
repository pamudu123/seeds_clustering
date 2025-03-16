import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.colors as mcolors
from matplotlib.patches import Patch

def barplot_prediction_correctness(y_true:np.ndarray, y_pred:np.ndarray, labels:np.ndarray=None, figsize:tuple=(8, 5), 
                            title:str="Correct vs. Incorrect Predictions per Actual Label", colors:tuple=('skyblue', 'salmon')):
    """
    Create a bar chart comparing correct and incorrect predictions for each class.
    
    Parameters:
    -----------
    y_true : array-like
        Ground truth labels
    y_pred : array-like
        Predicted labels
    labels : array-like, optional
        List of class labels to include in the plot. If None, unique values from y_true are used.
    figsize : tuple, optional
        Figure size as (width, height) in inches. Default is (8, 5).
    colors : tuple, optional
        Colors for (correct, incorrect) bars. Default is ('skyblue', 'salmon').
    title : str, optional
        Plot title. Default is 'Correct vs. Incorrect Predictions per Actual Label'.
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The created figure object
    ax : matplotlib.axes.Axes
        The created axes object
    """
    # If no labels provided, use unique values from y_true
    if labels is None:
        labels = np.unique(y_true)
    
    # Calculate counts for correct and incorrect predictions per label
    correct_counts = []
    incorrect_counts = []
    for label in labels:
        correct = np.sum((y_true == label) & (y_pred == label))
        incorrect = np.sum((y_true == label) & (y_pred != label))
        correct_counts.append(correct)
        incorrect_counts.append(incorrect)
    
    # Set up the positions for the bars on the x-axis
    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars
    
    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)
    rects1 = ax.bar(x - width/2, correct_counts, width, label='Correct', color=colors[0])
    rects2 = ax.bar(x + width/2, incorrect_counts, width, label='Incorrect', color=colors[1])
    
    # Add text for labels, title, and custom x-axis tick labels
    ax.set_xlabel('Actual Label')
    ax.set_ylabel('Count')
    if title is None:
        title = 'Correct vs. Incorrect Predictions per Actual Label'
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    
    # Add value labels above each bar
    def autolabel(rects):
        """Attach a text label above each bar displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
    
    autolabel(rects1)
    autolabel(rects2)
    
    plt.tight_layout()
    
    return fig, ax


def plot_pca_visualizations(X, y_true:np.ndarray, y_pred:np.ndarray, class_labels:np.ndarray=None, display_size:tuple=(10, 5)):
    """
    Plot two PCA visualizations of the data:
    1. Points colored by predicted class
    2. Points colored by prediction correctness (correct/wrong)
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        The input features (7-dimensional)
    y_true : array-like, shape (n_samples,)
        The true class labels
    y_pred : array-like, shape (n_samples,)
        The predicted class labels
    class_labels : list or None
        List of class labels (if None, will use unique values from y_true)
    figsize : tuple, default=(16, 7)
        Figure size (width, height)
    
    Returns:
    --------
    fig : matplotlib Figure
        The figure containing both visualizations
    """
    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply PCA to reduce to 2 dimensions
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=display_size)
    
    # Set up colormap for classes
    if class_labels is None:
        class_labels = np.unique(y_true)
    
    num_classes = len(class_labels)
    class_cmap = plt.cm.get_cmap('tab10', num_classes)
    
    # Create a mapping from class labels to indices
    label_to_idx = {label: i for i, label in enumerate(class_labels)}
    
    # Plot 1: Color by predicted class
    scatter1 = ax1.scatter(X_pca[:, 0], X_pca[:, 1], 
                          c=[label_to_idx[label] for label in y_pred], 
                        #   cmap=class_cmap, 
                          alpha=0.7, 
                          s=50,
                          edgecolor='k',
                          linewidth=0.5)
    
    # Add title and labels
    ax1.set_title('PCA Visualization (Colored by Predicted Class)', fontsize=14)
    ax1.set_xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]:.2%} variance)', fontsize=12)
    ax1.set_ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]:.2%} variance)', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Add a legend for class labels
    legend_elements1 = [Patch(facecolor=class_cmap(label_to_idx[label]), 
                             edgecolor='k', 
                             label=f'Class {label}') 
                       for label in class_labels]
    ax1.legend(handles=legend_elements1, title="Predicted Classes", 
              loc="best", fontsize=10)
    
    # Plot 2: Color by correctness
    is_correct = y_pred == y_true
    # print(is_correct)
    
    # Define colors for correct and incorrect predictions
    correctness_colors = ['#e74c3c','#2ecc71']  # Green for correct, red for incorrect
    
    # Create a scatter plot
    scatter2 = ax2.scatter(X_pca[:, 0], X_pca[:, 1], 
                          c=is_correct.astype(int), 
                          cmap=mcolors.ListedColormap(correctness_colors), 
                          alpha=0.7, 
                          s=50,
                          edgecolor='k',
                          linewidth=0.5)
    
    # Add title and labels
    ax2.set_title('PCA Visualization (Colored by Prediction Correctness)', fontsize=14)
    ax2.set_xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]:.2%} variance)', fontsize=12)
    ax2.set_ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]:.2%} variance)', fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # Add a legend for correctness
    legend_elements2 = [
        Patch(facecolor=correctness_colors[1], edgecolor='k', label='Correct'),
        Patch(facecolor=correctness_colors[0], edgecolor='k', label='Incorrect')
    ]
    ax2.legend(handles=legend_elements2, title="Prediction Correctness", 
              loc="best", fontsize=10)
    
    # Adjust layout
    plt.tight_layout()
    
    # Add information about explained variance
    total_variance = pca.explained_variance_ratio_[0] + pca.explained_variance_ratio_[1]
    fig.suptitle(f'PCA Projection from 7D to 2D (Total Explained Variance: {total_variance:.2%})', 
                fontsize=16, y=1.05)
    
    return fig, pca