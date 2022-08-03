# Install imate with ;#"\texttt{pip install imate}"#;
from numpy import logspace
import imate

# Generate a sample correlation matrix using the kernel ;#$e^{-r/0.1}$#;.
A = imate.sample_matrices.correlation_matrix(
        50, dimension=2, kernel='exponential', scale=0.1) ;# \label{ln:A} #;

# Create an interpolating object for
# ;#$f_p : t \mapsto \| \mathbf{A} + t \mathbf{I} \|_p$, $p=-1$#;.
f = imate.InterpolateSchatten(A, B=None, p=-1, ti=logspace(-4, 3, 8), kind='IMBF',
                                 options={'method': 'cholesky'}) ;# \label{ln:tau} #;

# Interpolate 1000 points in ;#$[10^{-4}, 10^{3}]$#;.
t = logspace(-4, 3, 1000)
y = f(t) ;# \label{ln:f} #;

# Plot the interpolated normalized curve ;#$ \tau_p(t) = \| \mathbf{A} + t \mathbf{I} \|_p / \| \mathbf{I} \|_p  $#;, compare with exact values.
f.plot(t, compare=True, normalize=True)  ;# \label{ln:plot} #;
