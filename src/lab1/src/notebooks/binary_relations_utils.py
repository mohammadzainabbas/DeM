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

def CompleteOrderCheck(graph: Union[nx.Graph, npmat.matrix]) -> bool:
    """
    Checks if the graph/matrix is a complete order
    """
    return CompleteCheck(graph) and AntisymmetricCheck(graph) and TransitiveCheck(graph)

def CompletePreOrderCheck(graph: Union[nx.Graph, npmat.matrix]) -> bool:
    """
    Checks if the graph/matrix is a complete preorder
    """
    return CompleteCheck(graph) and TransitiveCheck(graph)

def StrictRelation(graph: Union[nx.Graph, npmat.matrix]) -> npmat.matrix:
    """
    Returns a strict relation of a given graph/matrix
    """
    matrix = graph_to_matrix(graph)
    output = matrix.copy()

    for i in range(0,len(matrix)):
        for j in range(0,len(matrix)):
            if matrix[i, j] == 1 and matrix[j, i] == 1:
                output[i, j], output[j, i] = 0, 0
    return npmat.asmatrix(output)

def IndifferenceRelation(graph: Union[nx.Graph, npmat.matrix]) -> npmat.matrix:
    """
    Returns an indifference relation of a given graph/matrix
    """
    matrix = graph_to_matrix(graph)
    output = np.zeros(matrix.shape)
    for i in range(0,len(matrix)):
        for j in range(0,len(matrix)):
            if matrix[i, j] == 1 and matrix[j, i] == 1:
                output[i, j], output[j, i] = 1, 1
    return npmat.asmatrix(output)

def Topologicalsorting(graph: Union[nx.Graph, npmat.matrix]) -> list:
    """
    Returns a topological sort of a given graph/matrix
    """
    
    def dagCheck(matrix: npmat.matrix) -> bool:
        matrix -= np.diag(np.diagonal(matrix))
        matrix_reachable, matrix_reachable_sum = np.identity(len(matrix)), np.zeros(matrix.shape)
        for i in range(0, len(matrix)):
            matrix_reachable = matrix_reachable.dot(matrix)
            matrix_reachable_sum += matrix_reachable
        return np.all(np.diagonal(matrix_reachable_sum.dot(matrix_reachable_sum)) == 0)
    
    matrix = graph_to_matrix(graph)

    if not dagCheck(matrix):
        print_error("Graph is not a DAG")
        return
    
    topologicalSorting_matrix = matrix - np.diag(np.diagonal(matrix))
    topologicalSorting_list, original_list = [], list(range(0,len(matrix)))
    while len(original_list) != 0:
        sum = topologicalSorting_matrix.sum(axis=1)
        for i in original_list:
            if sum[i] == 0:
                topologicalSorting_list.append(i)
                topologicalSorting_matrix[:,i] = 0
                original_list.remove(i)
    return topologicalSorting_list