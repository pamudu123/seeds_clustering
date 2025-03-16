import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score, precision_score, recall_score

def purity_score(y_true, y_pred):
    """
    Calculate purity score for clustering evaluation.
    
    Parameters:
    -----------
    y_true : array-like
        Ground truth labels
    y_pred : array-like
        Predicted cluster labels
        
    Returns:
    --------
    float
        Purity score (range: 0 to 1, higher is better)
    """
    # Compute the confusion matrix
    contingency_matrix = confusion_matrix(y_true, y_pred)
    # Sum the maximum count for each cluster
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)


def calculate_classification_metrics(y_true, y_pred, print_output=True):
    """
    Calculate and optionally print various classification metrics.
    
    Parameters:
    -----------
    y_true : array-like
        Ground truth labels
    y_pred : array-like
        Predicted labels
    print_output : bool, default=True
        Whether to print the metrics to console
        
    Returns:
    --------
    dict
        Dictionary containing all calculated metrics:
        - 'report': Classification report (string)
        - 'accuracy': Accuracy score (float)
        - 'confusion_matrix': Confusion matrix (array)
        - 'f1_score': F1 score (float)
        - 'precision': Precision score (float)
        - 'recall': Recall score (float)
        - 'purity': Purity score (float)
    """
    # Calculate classification report
    report = classification_report(y_true, y_pred)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_true, y_pred)
    
    # Calculate confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred)
    
    # Calculate F1 score
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    # Calculate precision
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # Calculate recall
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # Calculate purity score
    purity = purity_score(y_true, y_pred)
    
    # Store all metrics in a dictionary
    metrics = {
        'report': report,
        'accuracy': accuracy,
        'confusion_matrix': conf_matrix,
        'f1_score': f1,
        'precision': precision,
        'recall': recall,
        'purity': purity
    }
    
    # Print metrics if requested
    if print_output:
        print(report)
        print(f"Accuracy: {accuracy:.4f}")
        print("Confusion Matrix:")
        print(conf_matrix)
        print(f"F1 Score: {f1:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"Purity Score: {purity:.4f}")
    
    return metrics

def plot_confusion_matrix(y_true:np.ndarray, y_pred:np.ndarray, class_labels:np.ndarray=None, display_size:tuple=(6, 4)):
    """
    Plot a confusion matrix as a heatmap.
    
    Parameters:
    -----------
    y_true : array-like
        Ground truth labels
    y_pred : array-like
        Predicted labels
    class_labels : array-like, optional
        List of class labels to display. If None, unique values from y_true are used.
    """
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Define class labels if not provided
    if class_labels is None:
        class_labels = np.unique(y_true)
    
    # Create the plot
    plt.figure(figsize=display_size)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    
    # Set tick marks and labels
    tick_marks = np.arange(len(class_labels))
    plt.xticks(tick_marks, class_labels)
    plt.yticks(tick_marks, class_labels)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    
    # Add text annotations for each cell
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.show()


def plot_prediction_accuracy_by_class(y_true:np.ndarray, y_pred:np.ndarray, labels:np.ndarray=None, display_size:tuple=(8, 5)):
    """
    Plot a bar chart showing correct and incorrect predictions for each class.
    
    Parameters:
    -----------
    y_true : array-like
        Ground truth labels
    y_pred : array-like
        Predicted labels
    labels : array-like, optional
        List of class labels to include in the plot. If None, unique values from y_true are used.
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
    x = np.arange(len(labels))
    width = 0.35
    
    # Create the plot
    fig, ax = plt.subplots(figsize=display_size)
    rects1 = ax.bar(x - width/2, correct_counts, width, label='Correct', color='skyblue')
    rects2 = ax.bar(x + width/2, incorrect_counts, width, label='Incorrect', color='salmon')
    
    # Add labels and title
    ax.set_xlabel('Actual Label')
    ax.set_ylabel('Count')
    ax.set_title('Correct vs. Incorrect Predictions per Actual Label')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    
    # Add value labels on bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')
    
    autolabel(rects1)
    autolabel(rects2)
    
    plt.tight_layout()
    plt.show()