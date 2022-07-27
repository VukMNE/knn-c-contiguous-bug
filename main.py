import numpy as np
import pickle

from sklearn.neighbors import KNeighborsRegressor

file = open('knn.pkl', 'rb')
knn_model: KNeighborsRegressor = pickle.load(file)
knn_model.predict(np.array([175.6, 12.2, 97.8, 4.9, 92.6, -0.4999999999999998,
                            0.8660254037844387, -1.0, 1.2246467991473532e-16,
                            65.0, 83.0, 29.0,0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).reshape(1, -1))