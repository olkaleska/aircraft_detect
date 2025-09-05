"""
основний робочий код
містить зчитування фото
обробку
pca
svm
тести
"""
import cv2
import numpy as np
from PIL import Image
from sympy import Matrix, symbols, eye
from sympy import solve
import os
import random

def read_picture(path_to_jpg):
    """
    Ця функція зчитує шлях до зображення, перетворює його у чб
    зменшує розмірність, переводить у матрицю, а далі у арей і нормалізує
    
    : повертає нормалізований "полегшений" арей зображення з шляху
    """
    img = Image.open(path_to_jpg).convert('L')
    new_size = (100, 70)
    img_resized = img.resize(new_size)
    arr = np.array(img_resized)
    flat_img = arr.flatten()
    flat_img = flat_img / 255.0

    return flat_img.tolist()

def find_mean(list_of_arrays):
    """
    Приймаємо ліст векторів зображень, шукаємо для них mean та центруємо всі вектори
    """
    n = len(list_of_arrays)
    lengthh = len(list_of_arrays[0])

    res = [sum(array[i] for array in list_of_arrays) / n for i in range(lengthh)]
    for arrays in list_of_arrays:
        for i in range(lengthh):
            arrays[i] -= res[i]

    print('Середнє жнайдено, матриці процентровано')
    print('Func find_mean was successfully done')
    return res

def egenvectors(meaned_list_of_arrays):
    """
    PCA: Повертає матрицю Z (всі зображення, спроєктовані на 50 головних компонент)
    та матрицю V_k — 50 власних векторів для подальшої проєкції.
    """
    A = np.array(meaned_list_of_arrays)
    # print(1)
    covariance_matrix = np.dot(A.T, A) / A.shape[0]
    print(2)
    eigvals, eigvecs = np.linalg.eigh(covariance_matrix)
    print(3)
    sorted_indices = np.argsort(eigvals)[::-1][:30] # берем перші 30
    print(4)
    V_k = eigvecs[:, sorted_indices] # d × 30
    print(5)
    Z = np.dot(A, V_k)  # (n_psc * 15000) × (15000 * 30) = n_psc × 30

    print('Fast egenvectors function done.')
    return Z, V_k


class SVM:
    """
    Наш svm
    """
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X, y):
        """
        Навчаємо 
        X -- матриця з 
        y -- ліст з 0 та 1 -- є ч нема літака

        J = lambda * ||w||^2 + sum[max(0, 1 - y(w * x -b))]/n
        If y * f(x) >= 1:
            J = lambda * ||w||^2
            J`w = 2 * lambda * w
            J`b = 0
        else:
            J = lambda * ||w||^2 + 1 - y(w * x -b)
            J`w = 2 * lambda * w - y * x
            J`b = y

        For each upd:
        w = w - a * dw
        b = b - a * db
        """
        n_samples, n_features = X.shape

        y = np.array(y)
        y_ = np.where(y <= 0, -1, 1)

        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                # тут це розділення на Ю Б 1
                if condition:
                    # w = w - a * dw
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    # w = w - a * dw
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y_[idx]))
                    # b = b - a * db
                    self.b -= self.lr * y_[idx]
        print('Func fit was successfully done')

    def predict(self, X):
        """
        Предікт -- передаємо матрицю зображення
        """
        approx = np.dot(X, self.w) - self.b
        print('Func predict was successfully done')
        return np.sign(approx)


def load_all_images(folder, rozsh):
    """
    стягуємо файли з папок
    """
    files = sorted([os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(rozsh)])
    return [read_picture(p) for p in files]


X0 = load_all_images('no_plane', '.png')     # клас 0
X1 = load_all_images('one_plane', '.jpg')    # клас 1

X_all = X0 + X1
mean_vector = find_mean(X_all)
Z, V_k = egenvectors(X_all)

y = [0] * len(X0) + [1] * len(X1) # формуємо 0 і 1 послідовно

print("X0:", len(X0))
print("X1:", len(X1))
print("X_all:", len(X_all))
print("y:", len(y))
print("Z:", Z.shape)

svm = SVM()
print('SVM created')
svm.fit(Z, y)

print('Almost end')

def classify_new_image(path, mean_vector, V_k, svm):
    """
    Нова фоточка на чек
    """
    x = read_picture(path)
    print(len(x))
    x_centered = np.array(x) - np.array(mean_vector)
    x_proj = np.dot(V_k.T, x_centered)
    prediction = svm.predict(x_proj.reshape(1, -1))
    return prediction[0]


# print("Клас:", classify_new_image('one_to_check/SUV_083.jpg', mean_vector, V_k, svm))  # Має бути 1
# print("Клас:", classify_new_image('plane_from_coords/a_tile_3.png', mean_vector, V_k, svm))  # Має бути 1
# print("Клас:", classify_new_image('plane_from_coords/a_tile_2.png', mean_vector, V_k, svm))  # Має бути 1
# print("Клас:", classify_new_image('plane_from_coords/a_tile_0.png', mean_vector, V_k, svm))  # Має бути -1

# def run_tests(plane_dir, no_plane_dir, mean_vector, V_k, svm):
#     image_paths = []
#     labels = []

#     # Збираємо зображення з літаками
#     for fname in os.listdir(plane_dir):
#         image_paths.append(os.path.join(plane_dir, fname))
#         labels.append(1)

#     # Збираємо зображення без літаків
#     for fname in os.listdir(no_plane_dir):
#         image_paths.append(os.path.join(no_plane_dir, fname))
#         labels.append(-1)

#     # Перемішуємо
#     combined = list(zip(image_paths, labels))
#     random.shuffle(combined)
#     image_paths, labels = zip(*combined)

#     correct = 0
#     total = len(labels)

#     for img_path, true_label in zip(image_paths, labels):
#         pred = classify_new_image(img_path, mean_vector, V_k, svm)
#         match = pred == true_label
#         print(f"{img_path} | Прогноз: {pred}, Очікуване: {true_label} | {'OK' if match else 'NO'}")
#         if match:
#             correct += 1

#     accuracy = correct / total
#     print(f"\nТочність: {accuracy:.2%} ({correct}/{total} правильно)")


# run_tests(
#     plane_dir='test/one_plane',
#     no_plane_dir='test/no_plane',
#     mean_vector=mean_vector,
#     V_k=V_k,
#     svm=svm
# )
