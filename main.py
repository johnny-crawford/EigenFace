import sklearn.datasets
import numpy as np
import matplotlib.pyplot as plt


def load_and_split_data(n_train_per_person=8):

    """
    Load Olivetti dataset and split into training/test sets.
    
    Parameters:
    -----------
    n_train_per_person : int
        Number of training images per person (default: 8)
        
    Returns:
    --------
    X_train : array, shape (320, 4096)
        Training face images
    X_test : array, shape (80, 4096)
        Test face images
    y_train : array, shape (320,)
        Training labels
    y_test : array, shape (80,)
        Test labels
    """

    faces = sklearn.datasets.fetch_olivetti_faces()

    training_indices = []
    for person_id in range(40): 
        for image_num in range(n_train_per_person): 
            training_index = person_id * 10 + image_num
            training_indices.append(training_index)

    test_indices = []
    for person_id in range(40):
        for image_num in range(n_train_per_person, 10): 
            test_index = person_id * 10 + image_num
            test_indices.append(test_index)

    X_train = faces.data[training_indices]
    X_test = faces.data[test_indices]

    y_train = faces.target[training_indices]
    y_test = faces.target[test_indices]

    return X_train, X_test, y_train, y_test


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
    predicted_person = y_train[closest_idx]

    predictions.append(predicted_person)

predictions = np.array(predictions)



