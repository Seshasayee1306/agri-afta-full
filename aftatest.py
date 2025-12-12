import numpy as np
import pandas as pd
y_all = df[config['target']].values
unique, counts = np.unique(y_all, return_counts=True)
print("Class distribution:", dict(zip(unique, counts)))

