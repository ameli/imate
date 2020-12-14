# ============
# Declarations
# ============

# Copy Vector
cdef void CopyVector(
        const double* InputVector,
        const int n,
        double* OutputVector) nogil

# Inner Product
cdef double InnerProduct(
        const double* Vector1,
        const double* Vector2,
        const int VectorLength) nogil

# Euclidean Norm
cdef double EuclideanNorm(
        const double* Vector,
        const int VectorLength) nogil

# Normalize Vector In Place
cdef double NormalizeVectorInPlace(
        double* Vector,
        const int VectorLength) nogil

# Normalize Vector And Copy
cdef double NormalizeVectorAndCopy(
        const double* Vector,
        const int VectorLength,
        double* OutputVector) nogil

# Dense Matrix Vector Multiplication
cdef void DenseMatrixVectorMultiplication(
        const double[:,::1] A,
        const double* b,
        const int NumRows,
        const int NumColumns,
        double* c) nogil

# Sparse Matrix Vector Multiplication
cdef void SparseMatrixVectorMultiplication(
        const double[:] A_data,
        const int[:] A_indices,
        const int[:] A_indptr,
        const double* b,
        const int NumRows,
        double* c) nogil

# Create Band Matrix
cdef void CreateBandMatrix(
        const double[:] alpha,
        const double[:] beta,
        const int NonZeroSize,
        const int TridiagonalFlag,
        double[:,::1] T) nogil
