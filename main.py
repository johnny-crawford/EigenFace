import sklearn.datasets
import numpy as np
import matplotlib.pyplot as plt


faces = sklearn.datasets.fetch_olivetti_faces()

training_indices = []
for person_id in range(40):  # 40 people
    for image_num in range(8):  # first 8 images
        training_index = person_id * 10 + image_num
        training_indices.append(training_index)

test_indices = []
for person_id in range(40):  # 40 people
    for image_num in range(8, 10):  # last 2 images
        test_index = person_id * 10 + image_num
        test_indices.append(test_index)

X_train = faces.data[training_indices]
X_test = faces.data[test_indices]

Y_train = faces.target[training_indices]
Y_test = faces.target[test_indices]

mean_face = X_train.mean(axis=0)


X_train_centred = X_train - mean_face 
X_test_centred = X_test - mean_face


C = np.dot(X_train_centred, np.transpose(X_train_centred))

eigenvalues, eigenvectors = np.linalg.eigh(C)
descending_indices = np.flip(np.argsort(eigenvalues))
sorted_eigenvalues = eigenvalues[descending_indices]
sorted_eigenvectors = eigenvectors[:, descending_indices]

n_eigenfaces = 110
eigenfaces = np.zeros((n_eigenfaces, 4096))

for i in range(n_eigenfaces):
    eigenvector = sorted_eigenvectors[:, i]

    eigenface = X_train_centred.T @ sorted_eigenvectors[:, i]

    eigenface = eigenface / np.linalg.norm(eigenface)

    eigenfaces[i] = eigenface



coefficients = X_train_centred @ eigenfaces.T



