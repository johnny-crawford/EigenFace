"""
Eigenface Face Recognition Implementation

This module implements the classical Eigenface algorithm for face recognition
using Principal Component Analysis (PCA) on the Olivetti faces dataset.

Author: Johnny Crawford
Date: 03/07/2025
"""

import sklearn.datasets
import numpy as np
import matplotlib.pyplot as plt

# Configuration constants
n_subjects = 40
n_images_per_subject = 10
image_size = (64, 64)
vector_size = 4096


def load_and_split_data(n_train_per_person=8):

    # Load Olivetti dataset and split into training/test sets.
    

    faces = sklearn.datasets.fetch_olivetti_faces()

    training_indices = []
    for person_id in range(n_subjects): 
        for image_num in range(n_train_per_person): 
            training_index = person_id * 10 + image_num
            training_indices.append(training_index)

    test_indices = []
    for person_id in range(n_subjects):
        for image_num in range(n_train_per_person, n_images_per_subject): 
            test_index = person_id * 10 + image_num
            test_indices.append(test_index)

    X_train = faces.data[training_indices]
    X_test = faces.data[test_indices]

    y_train = faces.target[training_indices]
    y_test = faces.target[test_indices]

    return X_train, X_test, y_train, y_test

def create_eigenfaces(X_train, n_components=50):
   
    # Compute eigenfaces using PCA on training data.

    mean_face = X_train.mean(axis=0)

    X_train_centred = X_train - mean_face 

    C = np.dot(X_train_centred, np.transpose(X_train_centred))

    eigenvalues, eigenvectors = np.linalg.eigh(C)

    descending_indices = np.flip(np.argsort(eigenvalues))
    sorted_eigenvalues = eigenvalues[descending_indices]
    sorted_eigenvectors = eigenvectors[:, descending_indices]

    eigenfaces = np.zeros((n_components, 4096))

    for i in range(n_components):
        eigenvector = sorted_eigenvectors[:, i]

        eigenface = X_train_centred.T @ sorted_eigenvectors[:, i]

        eigenface = eigenface / np.linalg.norm(eigenface)

        eigenfaces[i] = eigenface

    return eigenfaces, mean_face, sorted_eigenvalues[:n_components]


def project_faces(X, mean_face, eigenfaces):
    
    # Project face images onto eigenface space.

    X_centred = X - mean_face

    projections = X_centred @ eigenfaces.T

    return projections

def recognise_faces(test_projections, train_projections, y_train):
    predictions = []

    for i in range(len(test_projections)):
        test_proj = test_projections[i]

        distances = np.linalg.norm(train_projections - test_proj, axis=1)
        closest_idx = np.argmin(distances)
        predicted_person = y_train[closest_idx]

        predictions.append(predicted_person)

    return np.array(predictions)

def main():
    # Step 1: Load and split data
    print("Loading Olivetti faces dataset...")
    X_train, X_test, y_train, y_test = load_and_split_data()
    print(f"Loaded {len(X_train)} training and {len(X_test)} test images")
    
    # Step 2: Compute eigenfaces from training data only
    print("\nComputing eigenfaces...")
    eigenfaces, mean_face, eigenvalues = create_eigenfaces(X_train, n_components=50)
    print(f"Computed {len(eigenfaces)} eigenfaces")
    
    # Step 3: Project both training and test data
    print("\nProjecting faces onto eigenspace...")
    train_projections = project_faces(X_train, mean_face, eigenfaces)
    test_projections = project_faces(X_test, mean_face, eigenfaces)
    
    # Step 4: Perform face recognition
    print("\nPerforming face recognition...")
    predictions = recognise_faces(test_projections, train_projections, y_train)
    
    # Step 5: Evaluate results
    accuracy = np.mean(predictions == y_test)
    print(f"\nResults:")
    print(f"Overall accuracy: {accuracy * 100:.1f}%")
    print(f"Correctly recognized: {np.sum(predictions == y_test)} out of {len(y_test)} faces")


if __name__ == "__main__":
    main()


