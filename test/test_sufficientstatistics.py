import os
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
source = open('../platforms/cuda/kernels/sufficientstatistics.cu').read()
mod = SourceModule(source, no_extern_c=True, include_dirs=[os.path.abspath('../platforms/cuda/kernels/')])


N = 10000
nColsA = 4
nColsB = 32

A = np.random.randn(N, nColsA).astype(np.float32)
B = np.random.randn(N, nColsB).astype(np.float32)
C1 = np.zeros((nColsA, nColsB), dtype=np.float32)
C2 = np.zeros((nColsA, nColsB), dtype=np.float32)
D = np.zeros(nColsA, dtype=np.float32)

mod.get_function('sufficientstatistics')(
    cuda.In(A), cuda.In(B), np.int32(N), np.int32(nColsA), np.int32(nColsB),
    cuda.InOut(C1), cuda.InOut(C2), cuda.InOut(D),
    block=(128, 1, 1), grid=(1,1))

#print 'A\n', A
#print 'B\n', B
print 'C1\n', C1
print 'Ref C1\n', np.dot(A.T, B)
print '\nC2\n', C2
print 'Ref2\n', np.dot(A.T, np.square(B))
print '\nR-D', D
print 'C-D', np.sum(A, axis=0)

print '\nError Norms'
print np.linalg.norm(C1 - np.dot(A.T, B))
print np.linalg.norm(C2 - np.dot(A.T, np.square(B)))
print np.linalg.norm(D - np.sum(A, axis=0))
