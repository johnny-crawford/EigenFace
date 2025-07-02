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

# mean_face_2d = mean_face.reshape(64, 64)

# Display it
# plt.imshow(mean_face_2d, cmap='gray')
# plt.title('Average Face')
# plt.axis('off')
# plt.show()

# print(mean_face_2d.shape)

X_train_centred = X_train - mean_face 
X_test_centred = X_test - mean_face

# print("Mean of centered data:", X_train_centred.mean(axis=0).mean())

C = np.dot(X_train_centred, np.transpose(X_train_centred))

eigenvalues, eigenvectors = np.linalg.eigh(C)
descending_indices = np.flip(np.argsort(eigenvalues))
sorted_eigenvalues = eigenvalues[descending_indices]
sorted_eigenvectors = eigenvectors[:, descending_indices]

eigenface = X_train_centred.T @ sorted_eigenvectors[0]

magnitude = np.linalg.norm(eigenface)
normalised_eigenface = eigenface / magnitude


'''
normalised_eigenface_2d = normalised_eigenface.reshape(64, 64)
plt.imshow(normalised_eigenface_2d, cmap='gray')
plt.axis('off')
plt.show()
'''


# Calculate cumulative variance explained
total_variance = np.sum(sorted_eigenvalues)
cumulative_variance = np.cumsum(sorted_eigenvalues) / total_variance

# Find how many components for 95% variance
n_components = np.argmax(cumulative_variance >= 0.95) + 1
print(f"Need {n_components} eigenfaces for 95% variance")
