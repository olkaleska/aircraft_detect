import random
import os
from main import classify_new_image, read_picture, mean_vector, V_k, svm 

def test_accuracy_on_test_set():
    image_paths = []
    labels = []

    # 50 літаків
    for fname in os.listdir("test/one_plane"):
        image_paths.append(os.path.join("test/one_plane", fname))
        labels.append(1)

    # 50 без літаків
    for fname in os.listdir("test/no_plane"):
        image_paths.append(os.path.join("test/no_plane", fname))
        labels.append(-1)

    # перемішати
    combined = list(zip(image_paths, labels))
    random.shuffle(combined)
    image_paths, labels = zip(*combined)

    correct = 0
    for img, label in zip(image_paths, labels):
        pred = classify_new_image(img, mean_vector, V_k, svm)
        if pred == label:
            correct += 1

    acc = correct / len(labels)
    assert acc > 0.8, f"Accuracy too low: {acc}"

test_accuracy_on_test_set()