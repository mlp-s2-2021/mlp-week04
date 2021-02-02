import numpy as np
import pandas as pd

np.random.seed(1234)

n = 250

# Generate x locations between 0 and 1
x = np.sort(np.random.rand(n)).reshape(-1,1)

# Construct a covariance matrix using squared exponential cov, add a small nugget variance.
C = np.exp( -(75) * (x - x.T)**2 ) + np.diag([0.025]*n)

# Generate y draws from the MVN
y = np.linalg.cholesky(C) @ np.random.randn(n)


# Create and save a data frame
d = pd.DataFrame(data = {'x': x.flatten(), 'y': y})
d.to_csv("gp.csv")
