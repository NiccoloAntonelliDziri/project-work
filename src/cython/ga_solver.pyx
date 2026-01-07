# distutils: language = c++
# cython: boundscheck=False, wraparound=False, cdivision=True

import numpy as np
cimport numpy as np
from cython.parallel import prange
from libc.math cimport pow
from libc.stdlib cimport malloc, free

cdef double calculate_edge_cost(double dist, double beta_dist, double current_gold, double alpha, double beta) nogil:
    return dist + pow(alpha * current_gold, beta) * beta_dist

cdef double solve_split(long[:] permutation, double[:, :] dist_matrix, double[:, :] beta_dist_matrix, double[:] gold_values, double alpha, double beta, int n) nogil:
    cdef double* dp = <double*> malloc((n + 1) * sizeof(double))
    cdef int i, j
    cdef int next_customer
    cdef int current_node
    cdef double current_gold
    cdef double trip_cost
    cdef double return_cost
    cdef double total_cost
    cdef double dist_val
    cdef double beta_dist_val
    
    if dp == NULL:
        return -1.0 # Error

    dp[0] = 0.0
    for i in range(1, n + 1):
        dp[i] = 1.0/0.0 # Infinity

    for i in range(n):
        if dp[i] == 1.0/0.0:
            continue
            
        current_gold = 0.0
        trip_cost = 0.0
        current_node = 0 # Depot
        
        for j in range(i + 1, n + 1):
            next_customer = permutation[j-1]
            
            # Cost from current to next
            dist_val = dist_matrix[current_node, next_customer]
            beta_dist_val = beta_dist_matrix[current_node, next_customer]
            trip_cost += calculate_edge_cost(dist_val, beta_dist_val, current_gold, alpha, beta)
            
            current_gold += gold_values[next_customer]
            current_node = next_customer
            
            # Cost to return
            dist_val = dist_matrix[current_node, 0]
            beta_dist_val = beta_dist_matrix[current_node, 0]
            return_cost = calculate_edge_cost(dist_val, beta_dist_val, current_gold, alpha, beta)
            
            total_cost = dp[i] + trip_cost + return_cost
            
            if total_cost < dp[j]:
                dp[j] = total_cost
                
    cdef double result = dp[n]
    free(dp)
    return result

def evaluate_population_cython(long[:, :] population, double[:, :] dist_matrix, double[:, :] beta_dist_matrix, double[:] gold_values, double alpha, double beta):
    cdef int pop_size = population.shape[0]
    cdef int n = population.shape[1]
    cdef double[:] results = np.empty(pop_size, dtype=np.float64)
    cdef int i
    
    # Release GIL for parallelism
    with nogil:
        for i in prange(pop_size, schedule='dynamic'):
            results[i] = solve_split(population[i], dist_matrix, beta_dist_matrix, gold_values, alpha, beta, n)
            
    return np.asarray(results)
