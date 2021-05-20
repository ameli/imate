# ============
# Declarations
# ============

cdef double matern_kernel(
        const double x,
        const double correlation_scale,
        const double nu) nogil

cdef double euclidean_distance(
        const double[:] point1,
        const double[:] point2,
        const int dimension) nogil
