import numpy as np


# vel = vel + (c1 * rand() * (pos_best - pos)) + (c2*rand()*(pos_global_best - pos))
# pos = pos + vel


MATRIX = np.array([
    [1, 4, 40, 10],
    [2, 7, 42, 6],
    [3, 5, 25, 5],
    [4, 3, 12, 4]])

TOTAL_PROFIT = 0
TOTAL_WEIGHT = 0
CAPACITY = 10

def default(matrix):
    """
    Select elements by the order given in the matrix
    Returns:
        number: weight, profit, max element index
    """
    return matrix[1, 1], matrix[1, 2], 1

def max_profit(matrix):
    """
    Select element with biggest profit
    Returns:
        number: weight, profit, max element index
    """
    index_max_elem = np.argmax(matrix[:, 2], axis=0)
    weight = matrix[index_max_elem, 1]
    profit = matrix[index_max_elem, 2]
    return weight, profit, index_max_elem

def min_profit(matrix):
    """
    select element with smallest profit
    Returns:
        number: weight, profit, min element index
    """
    index_min_weight = np.argmin(matrix[:,1], axis=0)
    weight = matrix[index_min_weight, 1]
    profit = matrix[index_min_weight, 2]
    return weight, profit, index_min_weight

def max_profit_weight(matrix):
    """
    select element with best profit/weight
    Returns:
        number: weight, profit, max element index
    """
    index_max_elem = np.argmax(matrix[:, 3], axis=0)
    weight = matrix[index_max_elem, 1]
    profit = matrix[index_max_elem, 2]
    return weight, profit, index_max_elem

def get_avg_profit(matrix):
    """
    Returns: normalized average profit
    """
    max_val = max(matrix[:, 2])
    avg_profit = np.mean(matrix[:, 2]) * (1/max_val)
    return np.array([avg_profit])

# set of simple heuristics
ACTIONS = [default, max_profit, min_profit, max_profit_weight]

SELECTOR = np.array([
    [0.7, max_profit_weight],
    [0.6, max_profit]])

# euclidean distance can be calculated with dist = numpy.linalg.norm(a-b)
def kp_hyper_heuristic(selector, matrix, total_profit, total_weight, capacity):
    """
    Solves the KP using the avg profit characteritics and the rule values defined in selector
    Returns:
        total weight, total profit
    """
    while matrix.shape[0] != 0 and total_weight < capacity:
        f_t = get_avg_profit(matrix) # consider flexibility for adding more characteristics
        number_rules = selector.shape[0]
        dist_to_rules = []
        for i in range(number_rules):
            rule_values = selector[i, :-1]
            dist = np.linalg.norm(rule_values-f_t)
            dist_to_rules.append(dist)

        selected_rule = np.argmin(dist_to_rules)
        weight, profit, index = selector[selected_rule, -1](matrix)

        if total_weight + weight <= capacity:
            total_weight += weight
            total_profit += profit
        matrix = np.delete(matrix, index, axis=0)
    return total_weight, total_profit

print("Initial capacity: " + str(CAPACITY))
final_weight, final_profit = kp_hyper_heuristic(SELECTOR, MATRIX, TOTAL_PROFIT, TOTAL_WEIGHT, CAPACITY)
print("Final profit: " + str(final_profit))

"""
preguntas
1. Cuales son los algoritmos que ajustan mejor los valores de los pesos?

propuestas
* añadir clase de caracteristicas 
* añadir clase de reglas
* añadir clase de KP hyper heristica que este compuesta de clase reglas
* añadir esta clase en un modulo de Hiper Heuristicas
"""
