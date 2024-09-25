import numpy as np

C = np.array([[0.70, 0.50],[0.50, 0.68]])

vals, vecs = np.linalg.eig(C)
print(vals, vecs)