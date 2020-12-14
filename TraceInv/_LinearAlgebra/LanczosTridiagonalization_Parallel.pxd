# ============
# Declarations
# ============

cdef int LanczosTridiagonalization(
        const double[:,::1] A,
        const double[:] A_Data,
        const int[:] A_ColumnIndices,
        const int[:] A_IndexPointer,
        const double[:] v,
        const int n,
        const int m,
        const double Tolerance,
        double[:] alpha,
        double[:] beta) nogil
