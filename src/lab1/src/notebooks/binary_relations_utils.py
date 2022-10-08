import numpy as np
from numpy import matrixlib as npmat

import networkx as nx
from typing import Union

# ------------------------ #
# Helper binary relation
# ------------------------ #
def graph_to_matrix(G: Union[nx.Graph, npmat.matrix]) -> npmat.matrix:
    """
    Converts a graph to a matrix
    """
    return nx.to_numpy_matrix(G) if isinstance(G, nx.Graph) else G

def CompleteCheck(graph: Union[npmat.matrix, nx.Graph]) -> bool:
    """
    Checks if the matrix is complete
    """
    matrix = graph_to_matrix(graph)
    n_row, n_col = matrix.shape
    for i in range(n_row):
        for j in range(n_col):
            if matrix[i, j] == 0 or matrix[i, j] == 0:
                return False
    return True

def ReflexiveCheck(graph: Union[nx.Graph, npmat.matrix]) -> bool:
    """
    Checks if the graph/matrix is reflexive
    """
    matrix = graph_to_matrix(graph)
    return np.all(np.diagonal(matrix) == 1)

def AntisymmetricCheck(graph: Union[nx.Graph, npmat.matrix]) -> bool:
    """
    Checks if the graph/matrix is antisymmetric
    """
    matrix = graph_to_matrix(graph)
    n_row, n_col = matrix.shape
    for i in range(n_row):
        for j in range(n_col):
            if matrix[i, j] == 1 and matrix[j, i] == 1 and i != j:
                return False
    return True

def SymmetricCheck(graph: Union[nx.Graph, npmat.matrix]) -> bool:
    """
    Checks if the graph/matrix is symmetric
    """
    matrix = graph_to_matrix(graph)
    return np.all(matrix == matrix.T)

def AntisymmetricCheck(graph: Union[nx.Graph, npmat.matrix]) -> bool:
    """
    Checks if the graph/matrix is antisymmetric
    """
    matrix = graph_to_matrix(graph)
    matrix_sum = matrix + matrix.T
    check = matrix_sum - np.diag(np.diagonal(matrix_sum))
    return (np.logical_not(np.any(check == 2))).all()

def TransitiveCheck(graph: Union[nx.Graph, npmat.matrix]) -> bool:
    """
    Checks if the graph/matrix is transitive
    """
    matrix = graph_to_matrix(graph)
    return np.all(np.linalg.matrix_power(matrix, 3) == matrix)

def NegativetrasiitiveCheck(graph: Union[nx.Graph, npmat.matrix]) -> bool:
    """
    Checks if the graph/matrix is negative transitive
    """
    matrix = graph_to_matrix(graph)
    return np.all(np.linalg.matrix_power(matrix, 3) == 0)

