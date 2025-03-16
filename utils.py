import numpy as np

def transform_predictions(y_pred: np.ndarray, mapping: dict = {2: 1, 0: 3, 1: 2}) -> np.ndarray:
    """
    Transforms predictions based on a given mapping, with a default mapping provided.

    Parameters:
    -----------
    y_pred : np.ndarray
        Array of predicted labels to transform.
    mapping : dict, optional
        A dictionary specifying the mapping from old values to new values.
        Default is {2: 1, 0: 3, 1: 2}.

    Returns:
    --------
    np.ndarray
        Transformed array of predicted labels.
    """
    # Create a copy to avoid modifying the original array
    transformed_pred = y_pred.copy()

    # Apply the mapping rules
    for original, new in mapping.items():
        transformed_pred[y_pred == original] = new

    return transformed_pred