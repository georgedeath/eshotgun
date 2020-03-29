# cython: language_level=3

import cython
import numpy as np
cimport numpy as np

from .stats_package import mvnormcdf
from scipy.stats import norm

# use the c sqrt function instead of numpy's
cdef extern from "math.h":
    double sqrt(double m)
    
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def cy_get_full_cov(np.ndarray[np.float_t, ndim=2] K):
    
    cdef Py_ssize_t q = K.shape[0]
    cdef np.ndarray[np.float_t, ndim=3] cov = np.zeros((q, q, q), dtype='float')
    
    cdef Py_ssize_t i, j, k
    
    for i in range(q):
        for j in range(q):
            for k in range(q):
                if i == j and i == k:
                    cov[i, j, k] = K[i, i]
                    
                elif i == k:
                    cov[i, j, k] = K[i, i] - K[j, i]
                    
                elif i == j:
                    cov[i, j, k] = K[i, i] - K[k, i]
                    
                else:
                    cov[i, j, k] = K[j, k] + K[i, i] - K[j, i] - K[k, i]

    return cov
    
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def cy_get_mean_mat(np.ndarray[np.float_t, ndim=1] m_vector):
    
    cdef Py_ssize_t q = m_vector.shape[0]
    cdef np.ndarray[np.float_t, ndim=2] M = np.zeros((q, q), dtype='float')
    
    cdef Py_ssize_t i, j
    
    for i in range(q):
        for j in range(q):
            if i == j:
                M[i, j] = -m_vector[i]
            else:
                M[i, j] = m_vector[j] - m_vector[i]
                
    return M
    
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def cy_get_all_Ci(np.ndarray[np.float_t, ndim=2] B,   # shape q,q
                  np.ndarray[np.float_t, ndim=2] M,   # shape q,q
                  np.ndarray[np.float_t, ndim=3] cov, # shape q,q,q
                  np.float_t jitter=1e-8): 
    
    # number of batch points
    cdef Py_ssize_t q = B.shape[0]
    
    # indexing for loops
    cdef Py_ssize_t j, jidx, k, i
    
    # storage of final and itermediate results
    cdef np.ndarray[np.float_t, ndim=3] C = np.zeros((q, q, q-1))
    cdef np.ndarray[np.float_t, ndim=2] B_take_M = np.zeros((q, q))
    cdef np.float_t Kii_jitter
    
    # B - M
    for k in range(q):
        for j in range(q):
            B_take_M[k, j] = B[k, j] - M[k, j]
    
    # calculate Ci for each row of B and M
    for k in range(q):
        for i in range(q):
            Kii_jitter = cov[k, i, i] + jitter
            
            for jidx in range(q):
                if jidx == i:
                    continue

                if jidx > i:
                    j = jidx
                    jidx -= 1

                else:
                    j = jidx

                C[k, i, jidx] = B_take_M[k, j] - B_take_M[k, i] * cov[k, i, j] / Kii_jitter
    return C
    
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def cy_get_all_Ki(np.ndarray[np.float_t, ndim=3] cov,
                  np.float_t jitter=1e-8):
    # number of batch points
    cdef Py_ssize_t q = cov.shape[0]
    
    # indexing for loops
    cdef Py_ssize_t u, v, uidx, vidx
    
    # storage of final and itermediate results
    cdef np.ndarray[np.float_t, ndim=4] K = np.zeros((q, q, q-1, q-1))
    cdef np.float_t cov_ii_jitter
    
    for k in range(q):
        for i in range(q):
            
            cov_ii_jitter = cov[k, i, i] + jitter
            
            for uidx in range(q):
                if uidx == i:
                    continue

                if uidx > i:
                    u = uidx
                    uidx -= 1

                else:
                    u = uidx

                for vidx in range(q):
                    if vidx == i:
                        continue

                    if vidx > i:
                        v = vidx
                        vidx -= 1

                    else:
                        v = vidx

                    K[k, i, uidx, vidx] = cov[k, u, v] - cov[k, i, u] * cov[k, i, v] / cov_ii_jitter
            
    return K
    
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def cy_qEI(np.ndarray[np.float_t, ndim=2] m,
           np.ndarray[np.float_t, ndim=2] K,
           np.float_t incumbent,
           np.float_t obj_sense=-1):
    
    # number of batch points
    cdef Py_ssize_t q = m.shape[0]
    
    # indexing for loops
    cdef Py_ssize_t i, k
    
    cdef np.ndarray[np.float_t, ndim=1] mean = np.concatenate(m)
    cdef np.ndarray[np.float_t, ndim=1] meanT = mean - incumbent
    
    cdef np.ndarray[np.float_t, ndim=2] B = np.eye(q) * (-incumbent * obj_sense)
    
    cdef np.ndarray[np.float_t, ndim=2] M = obj_sense * cy_get_mean_mat(mean)
    
    cdef np.ndarray[np.float_t, ndim=3] cov = cy_get_full_cov(K)
    
    cdef np.ndarray[np.float_t, ndim=4] Ki = cy_get_all_Ki(cov)
    
    cdef np.ndarray[np.float_t, ndim=3] Ci = cy_get_all_Ci(B, M, cov)
    
    cdef np.float_t ei = 0
    
    cdef np.ndarray[np.float_t, ndim=1] zeros_like_Bk = np.zeros(q)
    cdef np.ndarray[np.float_t, ndim=1] zeros_like_Ci = np.zeros(q-1)

    
    for k in range(q):
        ei += obj_sense * meanT[k] * mvnormcdf(B[k] - M[k], 
                                               zeros_like_Bk, 
                                               cov[k])
        for i in range(q):
            ei += (cov[k, i, k]
                   * norm.pdf(B[k, i], loc=M[k, i], scale=sqrt(cov[k,i,i]))
                   * mvnormcdf(Ci[k, i], zeros_like_Ci, Ki[k, i]))  
            
    return ei
