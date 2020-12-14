# =======
# Imports
# =======

from cython import boundscheck,wraparound
from libc.math cimport sqrt

# ===========
# Copy Vector
# ===========

@boundscheck(False)
@wraparound(False)
cdef void CopyVector(
        const double* InputVector,
        const int n,
        double* OutputVector) nogil:
    """
    Copies a vector to a new vector. Result is written in-place

    :param InputVector: A 1D array
    :type InputVector: c pointer

    :param n: Length of vector.
    :type n: int

    :param OutputVector: Output vector (written in place).
    :type OutputVector: c pointer
    """

    cdef int i
    for i in range(n):
        OutputVector[i] = InputVector[i]

# =============
# Inner Product
# =============

@boundscheck(False)
@wraparound(False)
cdef double InnerProduct(
        const double* Vector1,
        const double* Vector2,
        const int VectorLength) nogil:
    """
    Computes Euclidean inner product of two vectors.

    :param Vector1: 1D array
    :type Vector2: c pointer

    :param Vector2: 1D array
    :type Vector2: c pointer

    :param VectorLength: Length of vector
    :type VectorLength: int

    :return: Inner product of two vectors.
    :rtype: double
    """

    cdef int i
    cdef double InnerProd = 0.0
    
    for i in range(VectorLength):
        InnerProd += Vector1[i]*Vector2[i]

    return InnerProd

# ==============
# Euclidean Norm
# ==============

@boundscheck(False)
@wraparound(False)
cdef double EuclideanNorm(
        const double* Vector,
        const int VectorLength) nogil:
    """
    Computes the Euclidean norm of a 1D array.

    :param Vector: A pointer to 1D array
    :type Vector: double*

    :param VectorLength: Length of the array
    :type VectorLength: int

    :return: Euclidean norm
    :rtype: double
    """

    # Compute norm squared
    cdef double Norm2 = InnerProduct(Vector,Vector,VectorLength)

    # Norm
    Norm = sqrt(Norm2)

    return Norm

# =========================
# Normalize Vector In Place
# =========================

@boundscheck(False)
@wraparound(False)
cdef double NormalizeVectorInPlace(
        double* Vector,
        const int VectorLength) nogil:
    """
    Normalizes a vector based on Euclidean 2-norm. The result is written in-place.

    :param Vector: Input vector to be normalized in-place.
    :type Vector: c pointer

    :param VectorLength: Length of the input vector
    :type VectorLength: int

    :return: Norm
    :rtype: double
    """

    cdef int i

    # Norm of vector
    cdef double Norm = EuclideanNorm(Vector,VectorLength)

    # Normalize
    for i in range(VectorLength):
        Vector[i] = Vector[i] / Norm

    return Norm

# =========================
# Normalize Vector And Copy
# =========================

@boundscheck(False)
@wraparound(False)
cdef double NormalizeVectorAndCopy(
        const double* Vector,
        const int VectorLength,
        double* OutputVector) nogil:
    """
    Normalizes a vector based on Euclidean 2-norm. The result is written into another vector.

    :param Vector: Input vector.
    :type Vector: c pointer

    :param VectorLength: Length of the input vector
    :type VectorLength: int

    :param OutputVector: Output vector, which is the normalization of the input vector.
    :type OuptutVector: c pointer

    :return: Norm
    :rtype: double
    """

    cdef int i

    # Norm of vector
    cdef double Norm = EuclideanNorm(Vector,VectorLength)

    # Normalize
    for i in range(VectorLength):
        OutputVector[i] = Vector[i] / Norm

    return Norm

# ==================================
# Dense Matrix Vector Multiplication
# ==================================

@boundscheck(False)
@wraparound(False)
cdef void DenseMatrixVectorMultiplication(
        const double[:,::1] A,
        const double* b,
        const int NumRows,
        const int NumColumns,
        double* c) nogil:
    """
    Computes :math:`\\boldsymbol{c} = \mathbf{A} \\boldsymbol{b}` when :math:`\mathbf{A}` is dense.

    :param A: 2D dense array
    :type: cython.memoryview

    :param b: Column vector
    :type b: c pointer

    :param NumRows: Number of rows of ``A``
    :type NumRows: int

    :param NumColumns: Number of columns of ``A``
    :type NumColumns: int

    :param c: The output column vector (written in-place).
    :type c: c pointer
    """

    cdef int i,j

    for i in range(NumRows):
        c[i] = 0.0
        for j in range(NumColumns):
            c[i] = c[i] + A[i,j]*b[j]

# ===================================
# Sparse Matrix Vector Multiplication
# ===================================

@boundscheck(False)
@wraparound(False)
cdef void SparseMatrixVectorMultiplication(
        const double[:] A_Data,
        const int[:] A_ColumnIndices,
        const int[:] A_IndexPointer,
        const double* b,
        const int NumRows,
        double* c) nogil:
    """
    Computes :math:`\\boldsymbol{c} = \mathbf{A} \\boldsymbol{b}` when :math:`\mathbf{A}` is sparse and
    :math:boldsymbol{b}` is dense.

    The input matirx ``A`` should be a 
    `Compressed Sparse Row (CSR) <https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html>`_ 
    matrix.

    :param A_Data: CSR format data array of the sparse matrix. The length of this array is the nnz of the matrix.
    :type A_Data: cython.memoryview

    :param A_ColumnIndices: CSR format column indices of the sparse matrix. The length of this array is the nnz of the matrix.
    :type A_ColumnIndices: cython.memoryview

    :param A_IndexPointer: CSR format index pointer. The length of this array is one plus the number of rows of the matrix.
        Also, the first element of this array is ``0``, and the last element is the nnz of the matrix.
    :type A_IndexPointer: cython.memoryview

    :param b: Column vector with the length equal to the number of columns of ``A``.
    :type b: cython.memoryview

    :param NumRows: Number of rows of the matrix ``A``. This is essentially the size of ``A_IndexPointer`` array minus one.
    :type NumRows: int

    :param c: Output column vector with the same size as ``b``. This array is written in-place.
    :type c: cython.memoryview
    """

    cdef int Row
    cdef int IndexPointer
    cdef int Column

    for Row in range(NumRows):
        c[Row] = 0.0
        for IndexPointer in range(A_IndexPointer[Row],A_IndexPointer[Row+1]):
            Column = A_ColumnIndices[IndexPointer]
            c[Row] = c[Row] + A_Data[IndexPointer]*b[Column]

# ==================
# Create Band Matrix
# ==================

cdef void CreateBandMatrix(
        const double[:] alpha,
        const double[:] beta,
        const int NonZeroSize,
        const int TridiagonalFlag,
        double[:,::1] T) nogil:
    """
    Creates bi-diagonal or symmetric tri-diagonal matrix from the diagonal array (alpha) and off-diagonal array (beta).

    The output is written in place (in matrix ``T``). The output is only written up to the ``NonZeroSize`` element, 
    that is: ``T[:NonZeroSize,:NonZeroSize]`` is filled, and the rest is assumed to be zero.

    Depending on the ``TridiagonalFlag``, the matrix ``T`` is upper bi-diagonal or symmetric tri-diagonal.

    :param alpha: An array of length ``n``. All elements ``alpha[:]`` create the diagonals of matrix ``T``.
    :type alpha: cython.memoryview

    :param beta: An array of length ``n+1``. Elements ``beta[1:-1]`` create the upper off-diagonal of matrix ``T``, 
        making ``T`` an upper bi-diagonal matrix.
        In addition, if ``Tridiagonal`` is set to ``1``, the lower off-diagonal is also created similar to the
        upper off-diagonal, making ``T`` a symmetric tri-diagonal matrix.
    :type: cython.memoryview

    :param NonZeroSize: Up to the ``T[:NonZeroSize,:NonZeroSize]`` of the matrix ``T`` will be written. At most, 
        ``NonZeroSize`` can be ``n``, which is the length of ``alpha`` and the size of the square matrix ``T``. 
        If ``NonZerosize`` is less than ``n``, it is due to the fact that either ``alpha`` or ``beta`` has zero 
        elements after the ``Size`` element (possibly due to early termination of Lanczos iteration method).
    :type: int

    :param TridiagonalFlag: Boolean. If set to ``0``, the matrix ``T`` becomes upper bi-diagonal. If set to ``1``, 
        the matrix ``T`` becomes symmetric tri-diagonal.
    :type TridiagonalFlag: int

    :param T: A 2D  matrix (written in place) of the shape ``(n,n)``. This is the output of this function. 
        This matrix is assumed to be initialized to zero before calling this function.
    :type T: cython.memoryview
    """
 
    cdef int j

    for j in range(NonZeroSize):

        # Diagonals
        T[j,j] = alpha[j]

        # Off diagonals
        if j < NonZeroSize-1:

            # Upper off-diagonal
            T[j,j+1] = beta[j+1]

            # Lower off-diagonal, making tri-diagonal matrix
            if TridiagonalFlag:
                T[j+1,j] = beta[j+1]
