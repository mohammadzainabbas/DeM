import sys

# ------------------------ #
# Helper logging functions
# ------------------------ #
def print_log(text: str) -> None:
    """
    Prints the log
    """
    print(f"[ log ]: {text}")

def print_error(text: str) -> None:
    """
    Prints the error
    """
    print(f"[ error ]: {text}")

if __name__ == "__main__":
    print_error("This is a module. Please import it.")
    sys.exit(1)

import numpy as np
from numpy import matrixlib as npmat
import networkx as nx
from typing import Union

# ------------------------ #
# Helper matrix functions
# ------------------------ #
def get_matrix(n_row: int = 5, n_col: int = 5, low: int = 2, seed: int = 0) -> npmat.matrix:
    """
    Returns a random matrix of a given size

    Parameters
    ----------
    n_row: int
        Number of rows
    n_col: int
        Number of columns
    low: int
        Lower bound of the random numbers
    seed: int
        Seed for the random number generator
    
    Returns
    -------
    npmat.matrix
        A random matrix
    
    Example
    -------
    >>> get_matrix(n_row=n_row, n_col=n_col, seed=0)
    matrix([[0, 1, 1, 0, 1],
            [1, 1, 1, 1, 1],
            [1, 0, 0, 1, 0],
            [0, 0, 0, 0, 1],
            [0, 1, 1, 0, 0]])
    """
    np.random.seed(seed=seed)
    return npmat.asmatrix(np.random.randint(low, size=(n_row, n_col)))

def get_matrix_from_list(list_of_lists: list) -> npmat.matrix:
    """
    Returns a matrix from a list of lists

    Parameters
    ----------
    list_of_lists: list
        A list of lists
    
    Returns
    -------
    npmat.matrix
        A matrix
    
    Example
    -------
    >>> get_matrix_from_list([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    matrix([[1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]])
    """
    return npmat.asmatrix(list_of_lists)

def numpy_matrix_to_network_graph(matrix: npmat.matrix) -> nx.Graph:
    """
    Prints a network graph from a numpy matrix

    Parameters
    ----------
    matrix: npmat.matrix
        A numpy matrix
    
    Returns
    -------
    nx.Graph
        A network graph
    """
    return nx.from_numpy_matrix(matrix)

def plot_graph(graph: Union[npmat.matrix, nx.Graph]) -> None:
    """
    Prints a matrix

    Parameters
    ----------
    matrix: npmat.matrix
        A numpy matrix
    """
    if isinstance(graph, npmat.matrix): graph = numpy_matrix_to_network_graph(graph)
    nx.draw(graph, with_labels=True)
