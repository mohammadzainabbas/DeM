import numpy as np
from numpy import matrixlib as npmat

# ------------------------ #
# Helper binary relation
# ------------------------ #
def CompleteCheck(matrix: npmat.matrix) -> bool:
    """
    Checks if the matrix is complete
    """
    n_row, n_col = matrix.shape
    for i in range(n_row):
        for j in range(n_col):
            if matrix[i, j] == 0 or matrix[i, j] == 0:
                return False
    return True

def ReflexiveCheck(matrix: npmat.matrix) -> bool:
    """
    Checks if the matrix is reflexive
    """
    n_row, n_col = matrix.shape
    for i in range(n_row):
        if matrix[i, i] == 0:
            return False
    return True