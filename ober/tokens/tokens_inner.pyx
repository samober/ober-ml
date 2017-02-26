# distutils: extra_compile_args = -fopenmp
# distutils: extra_link_args = -fopenmp
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

import cython
import numpy as np
cimport numpy as np

from libc.stdio cimport *
from libc.stdlib cimport malloc, free
from libc.string cimport memset, memcpy
import scipy.linalg.blas as fblas

from cython.parallel cimport parallel, prange

import time
import codecs

REAL = np.float32
ctypedef np.float32_t REAL_t
INT = np.int32
ctypedef np.int32_t INT_t

cdef extern from "voidptr.h":
	void* PyCObject_AsVoidPtr(object obj)
	
ctypedef void (*sgemv_ptr) (const char *trans, const int *m, const int *n, const float *alpha, const float *A, const int *lda, const float *X, const int *incX, const float *beta, const float *Y, const int *incY) nogil
cdef sgemv_ptr sgemv=<sgemv_ptr>PyCObject_AsVoidPtr(fblas.sgemv._cpointer)

ctypedef void (*sgemm_ptr) (const char *transA, const char *transB, const int *l, const int *n, const int *m, const float *alpha, const float *A, const int *lda, const float *B, const int *ldb, const float *beta, const float *C, const int *ldc) nogil
cdef sgemm_ptr sgemm=<sgemm_ptr>PyCObject_AsVoidPtr(fblas.sgemm._cpointer)

cdef int ONE = 1
cdef REAL_t ONEF = <REAL_t>1.0
cdef REAL_t ZEROF = <REAL_t>0.0
cdef char TRANS = 'T'
cdef char NO_TRANS = 'N'

cdef struct GraphEdge:
	INT_t from_index
	INT_t to_index
	REAL_t weight
	
cdef void blas_dot(int m, int n, REAL_t *A, REAL_t *X, REAL_t *C) nogil:
	sgemv(&NO_TRANS, &m, &n, &ONEF, A, &m, X, &ONE, &ZEROF, C, &ONE)
	
# l -> rows of matrix C (number of vectors we are dotting)
# n -> columns of matrix C (total vocab count)
# m -> intermediate dim (vector size)
cdef void blas_dotm(int l, int n, int m, REAL_t *A, REAL_t *B, REAL_t *C) nogil:
	sgemm(&NO_TRANS, &TRANS, &l, &n, &m, &ONEF, A, &l, B, &n, &ZEROF, C, &l)

def export_distributed_graph(np.ndarray[REAL_t, ndim=2] n_vectors, filename, int n=200, int batch_size=500):
	# format filename
	filename_byte_string = filename.encode("UTF-8")
	cdef char *fname = filename_byte_string
	cdef FILE *outfile
	
	cdef int i, j
	cdef int N_TOKENS = n_vectors.shape[0], N_DIM = n_vectors.shape[1]
	# c array of vectors
	n_vectors = np.asfortranarray(n_vectors)
	cdef REAL_t *c_vectors = <REAL_t*>np.PyArray_DATA(n_vectors)
	# array of graph edges
	cdef GraphEdge *edges = <GraphEdge *>malloc(N_TOKENS * n * sizeof(GraphEdge))
	
	cdef int num_batches = N_TOKENS / batch_size + 1
	
	# template for nearest array
	cdef INT_t *nearest_temp = <INT_t*>malloc(N_TOKENS * sizeof(INT_t))
	for i in range(N_TOKENS):
		nearest_temp[i] = i
	
	# local buffers
	cdef REAL_t *c_similar
	cdef REAL_t *c_sim
	cdef REAL_t *c_vector
	cdef INT_t *c_nearest
	
	try:
		# loop through all tokens
		start_time = time.time()
		with nogil, parallel():
			c_similar = <REAL_t*>malloc(batch_size * N_TOKENS * sizeof(REAL_t))
			c_sim = <REAL_t*>malloc(N_TOKENS * sizeof(REAL_t))
			c_vector = <REAL_t*>malloc(batch_size * N_DIM * sizeof(REAL_t))
			c_nearest = <INT_t*>malloc(N_TOKENS * sizeof(INT_t))
			for i in prange(num_batches):
				calculate_edges(i, c_vectors, c_similar, c_sim, c_vector, c_nearest, edges, nearest_temp, N_TOKENS, N_DIM, n, batch_size)
			free(c_similar)
			free(c_sim)
			free(c_vector)
			free(c_nearest)
		print "Completed calculations in %f seconds ..." % (time.time() - start_time)
		print "Writing to file ..."
		outfile = fopen(fname, "wb")
		fwrite(edges, sizeof(GraphEdge), N_TOKENS * n, outfile)
#		fwrite(edges, sizeof(GraphEdge), num_batches * batch_size * n, outfile)
		fclose(outfile)
	finally:
		free(edges)
		free(nearest_temp)
		
cdef void calculate_edges(int index, REAL_t *c_vectors, REAL_t *c_similar, REAL_t *c_sim, REAL_t *c_vector, INT_t *c_nearest, GraphEdge *edges, const INT_t *nearest_temp, int N_TOKENS, int N_DIM, int n, int batch_size) nogil:
	cdef int i, j, offset
	cdef int start_index = index * batch_size
	cdef int end_index = min((index+1)*batch_size, N_TOKENS)
	cdef INT_t to_index
	
	if start_index >= N_TOKENS:
		return
	
	# fill vector array
	for i in range(start_index, end_index):
		for j in range(N_DIM):
			c_vector[j * batch_size + (i - start_index)] = c_vectors[j * N_TOKENS + i] # remember: fortran order (last dimension first)
			
	# get cosine similarity dot products
	blas_dotm(batch_size, N_TOKENS, N_DIM, c_vector, c_vectors, c_similar)
	
	# loop through batch
	for i in range(min(batch_size, end_index-start_index)):
		# reset nearest to original order
		memcpy(c_nearest, nearest_temp, N_TOKENS * sizeof(INT_t))
		# copy over similar matrix
		for j in range(N_TOKENS):
			c_sim[j] = c_similar[j * batch_size + i]
		# partition nearest
		argpartition(c_nearest, c_sim, n+1, N_TOKENS) # use n+1 because we will remove the matching index
		# reset offset
		offset = 0
		# extract first n items in graphedges
		for j in range(n+1):
			to_index = c_nearest[j]
			# check if we found the matching index and adjust offset
			if to_index == start_index + i:
				offset = -1
			edges[(start_index+i)*n+j+offset].from_index = start_index + i
			edges[(start_index+i)*n+j+offset].to_index = to_index
			edges[(start_index+i)*n+j+offset].weight = c_sim[to_index]
	
cdef void argpartition(INT_t *nearest, REAL_t *similar, int k, int size) nogil:
	cdef int from_i = 0, to_i = size - 1
	cdef int r, w
	cdef INT_t temp
	cdef REAL_t mid
	while from_i < to_i:
		r = from_i
		w = to_i
		mid = similar[nearest[(r + w) / 2]]
		
		while r < w:
			if similar[nearest[r]] <= mid:
				temp = nearest[w]
				nearest[w] = nearest[r]
				nearest[r] = temp
				w -= 1
			else:
				r += 1
		
		if similar[nearest[r]] < mid:
			r -= 1
			
		if k <= r:
			to_i = r
		else:
			from_i = r + 1