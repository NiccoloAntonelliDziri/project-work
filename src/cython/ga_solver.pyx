# distutils: language = c++
# cython: boundscheck=False, wraparound=False, cdivision=True, nonecheck=False

import numpy as np
cimport numpy as np
from cython.parallel import prange
from libc.math cimport pow, INFINITY
from libc.stdlib cimport malloc, free

# Function to solve split and return predecessors for route reconstruction
# This is for a SINGLE individual
def solve_split_single(long[:] permutation, double[:, :] dist_matrix, double[:, :] beta_dist_matrix, double[:] gold_values, double alpha, double beta, int n):
    cdef double* dp = <double*> malloc((n + 1) * sizeof(double))
    cdef int* predecessor = <int*> malloc((n + 1) * sizeof(int))
    cdef int i, j
    cdef int next_customer
    cdef int current_node
    cdef double current_gold
    cdef double trip_cost
    cdef double return_cost
    cdef double total_cost
    cdef double dist_val
    cdef double beta_dist_val
    cdef double alpha_beta = pow(alpha, beta)
    
    # Pre-extract depot distances (column 0)
    cdef double[:] dist_to_depot = np.ascontiguousarray(dist_matrix[:, 0], dtype=np.float64)
    cdef double[:] beta_dist_to_depot = np.ascontiguousarray(beta_dist_matrix[:, 0], dtype=np.float64)
    
    if dp == NULL:
        return None, None
    if predecessor == NULL:
        free(dp)
        return None, None

    dp[0] = 0.0
    for i in range(1, n + 1):
        dp[i] = INFINITY
        predecessor[i] = -1

    for i in range(n):
        if dp[i] == INFINITY:
            continue
            
        current_gold = 0.0
        trip_cost = 0.0
        current_node = 0 # Depot
        
        for j in range(i + 1, n + 1):
            next_customer = permutation[j-1]
            
            # Cost from current to next
            dist_val = dist_matrix[current_node, next_customer]
            beta_dist_val = beta_dist_matrix[current_node, next_customer]
            
            if current_gold == 0:
                 trip_cost += dist_val
            else:
                 trip_cost += dist_val + alpha_beta * pow(current_gold, beta) * beta_dist_val
            
            current_gold += gold_values[next_customer]
            current_node = next_customer
            
            # Cost to return
            if current_gold == 0:
                 return_cost = dist_to_depot[current_node]
            else:
                 return_cost = dist_to_depot[current_node] + alpha_beta * pow(current_gold, beta) * beta_dist_to_depot[current_node]
            
            total_cost = dp[i] + trip_cost + return_cost
            
            if total_cost < dp[j]:
                dp[j] = total_cost
                predecessor[j] = i
                
    result_cost = dp[n]
    
    # Result predecessors to python list
    cdef int[:] pred_view = <int[:n+1]> predecessor
    pred_list = np.asarray(pred_view).copy()

    free(dp)
    free(predecessor)
    
    return result_cost, pred_list

cdef double solve_split(long[:] permutation, double[:, :] dist_matrix, double[:, :] beta_dist_matrix, double[:] gold_values, double[:] dist_to_depot, double[:] beta_dist_to_depot, double alpha, double beta, int n) nogil:
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
    cdef double alpha_beta = pow(alpha, beta)
    cdef double gold_factor
    
    if dp == NULL:
        return -1.0 

    dp[0] = 0.0
    for i in range(1, n + 1):
        dp[i] = INFINITY

    for i in range(n):
        if dp[i] == INFINITY:
            continue
            
        current_gold = 0.0
        trip_cost = 0.0
        current_node = 0 # Depot
        
        for j in range(i + 1, n + 1):
            next_customer = permutation[j-1]
            
            # Cost from current to next
            dist_val = dist_matrix[current_node, next_customer]
            beta_dist_val = beta_dist_matrix[current_node, next_customer]
            
            # Optimization: If gold is 0, pow is 0
            if current_gold == 0:
                 trip_cost += dist_val
            else:
                 trip_cost += dist_val + alpha_beta * pow(current_gold, beta) * beta_dist_val
            
            current_gold += gold_values[next_customer]
            current_node = next_customer
            
            # Cost to return (use cached depot distances passed as arguments)
            if current_gold == 0:
                 return_cost = dist_to_depot[current_node]
            else:
                 return_cost = dist_to_depot[current_node] + alpha_beta * pow(current_gold, beta) * beta_dist_to_depot[current_node]
            
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
    
    # Pre-extract depot distances (column 0) to avoid repeated memory access patterns or allocations inside loop
    # We create continuous arrays for CPU cache friendliness
    cdef double[:] dist_to_depot = np.ascontiguousarray(dist_matrix[:, 0], dtype=np.float64)
    cdef double[:] beta_dist_to_depot = np.ascontiguousarray(beta_dist_matrix[:, 0], dtype=np.float64)

    # Release GIL for parallelism
    with nogil:
        for i in prange(pop_size, schedule='dynamic'):
            # Pass the depot arrays
            results[i] = solve_split(population[i], dist_matrix, beta_dist_matrix, gold_values, dist_to_depot, beta_dist_to_depot, alpha,  beta, n)
            
    return np.asarray(results)
