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

n_eigenfaces = 50
eigenfaces = np.zeros((n_eigenfaces, 4096))

for i in range(n_eigenfaces):
    eigenvector = sorted_eigenvectors[:, i]

    eigenface = X_train_centred.T @ sorted_eigenvectors[:, i]

    eigenface = eigenface / np.linalg.norm(eigenface)

    eigenfaces[i] = eigenface



train_projections = X_train_centred @ eigenfaces.T
test_projections = X_test_centred @eigenfaces.T


predictions = []

for i in range(80):
    test_proj = test_projections[i]

    distances = np.linalg.norm(train_projections - test_proj, axis=1)
    closest_idx = np.argmin(distances)
    predicted_person = Y_train[closest_idx]

    predictions.append(predicted_person)

predictions = np.array(predictions)


accuracy = np.mean(predictions == Y_test)
print(f"Overall accuracy: {accuracy * 100:.1f}%")

# Confusion analysis
print(f"Correctly recognized: {np.sum(predictions == Y_test)} out of 80 faces")
print(f"Misrecognized: {np.sum(predictions != Y_test)} faces")

