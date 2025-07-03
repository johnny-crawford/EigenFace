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

def visualise_mean_face(mean_face, save=True):
    # Display and save the average face
    plt.figure(figsize=(6, 6))
    plt.imshow(mean_face.reshape(64, 64), cmap='gray')
    plt.title('Average Face', fontsize=16)
    plt.axis('off')
    
    if save:
        plt.savefig('results/mean_face.png', dpi=150, bbox_inches='tight')
    plt.show()

def visualise_eigenfaces(eigenfaces, n_show=20, save=True):
    # Display and save grid of eigenfaces
    n_rows = 4
    n_cols = 5
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 10))
    axes = axes.flatten()
    
    for i in range(min(n_show, len(eigenfaces))):
        eigenface_2d = eigenfaces[i].reshape(64, 64)
        axes[i].imshow(eigenface_2d, cmap='gray')
        axes[i].set_title(f'Eigenface {i+1}')
        axes[i].axis('off')
    
    plt.suptitle('Top 20 Eigenfaces', fontsize=16)
    plt.tight_layout()
    
    if save:
        plt.savefig('results/eigenfaces_grid.png', dpi=150, bbox_inches='tight')
    plt.show()

def main():
    # Step 1: Load and split data
    print("Loading Olivetti faces dataset...")
    X_train, X_test, y_train, y_test = load_and_split_data()
    print(f"Loaded {len(X_train)} training and {len(X_test)} test images")
    
    # Step 2: Create eigenfaces from training data only
    print("\nComputing eigenfaces...")
    eigenfaces, mean_face, eigenvalues = create_eigenfaces(X_train, n_components=50)
    print(f"Computed {len(eigenfaces)} eigenfaces")

    # Visualize mean face and eigenfaces
    print("\nVisualizing results...")
    visualise_mean_face(mean_face)
    visualise_eigenfaces(eigenfaces)
    
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

    # Show which people were hard to recognize
    print("\nPer-person accuracy:")
    for person in range(n_subjects):
        person_mask = (y_test == person)
        if np.any(person_mask):
            person_accuracy = np.mean(predictions[person_mask] == person)
            if person_accuracy < 1.0:
                print(f"  Person {person}: {person_accuracy*100:.0f}%")

if __name__ == "__main__":
    main()


