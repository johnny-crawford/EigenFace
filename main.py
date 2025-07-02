import sklearn.datasets
import numpy
import matplotlib.pyplot as plt


faces = sklearn.datasets.fetch_olivetti_faces()

training_indices = []
for person_id in range(40):  # 40 people
    for image_num in range(8):  # first 8 images
        index = person_id * 10 + image_num
        training_indices.append(index)

print(training_indices)