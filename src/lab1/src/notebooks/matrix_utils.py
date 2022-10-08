def CompleteCheck(matrix: npmat.matrix) -> bool:
    """
    Checks if the matrix is complete
    """
    for i in range(n_row):
        for j in range(n_col):
            if matrix[i, j] == 0 or matrix[i, j] == 0:
                return False
    return True