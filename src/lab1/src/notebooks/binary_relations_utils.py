import numpy as np
from numpy import matrixlib as npmat

import networkx as nx
from typing import Union

# ------------------------ #
# Helper binary relation
# ------------------------ #
def graph_to_matrix(G: nx.Graph) -> npmat.matrix:
    """
    Converts a graph to a matrix
    """
    return nx.to_numpy_matrix(G)

def CompleteCheck(graph: Union[npmat.matrix, nx.Graph]) -> bool:
    """
    Checks if the matrix is complete
    """
    if isinstance(graph, nx.Graph):
        matrix = nx.to_numpy_matrix(graph)
    else:
        matrix = graph
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