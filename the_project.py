"""
Мій дуже повільний варіант через старі штуки
"""
import cv2
import numpy as np
from PIL import Image
from sympy import Matrix, symbols, eye
from sympy import solve
import os

def read_picture(path_to_jpg):
    """
    Ця функція зчитує шлях до зображення, перетворює його у чб
    зменшує розмірність, переводить у матрицю, а далі у арей і нормалізує
    
    : повертає нормалізований "полегшений" арей зображення з шляху
    """
    img = Image.open(path_to_jpg).convert('L')
    new_size = (150, 100)  # (ширина, висота) - це приклад, обери свій розмір
    img_resized = img.resize(new_size)
    arr = np.array(img_resized)  # (H, W)
    flat_img = arr.flatten()
    flat_img = flat_img / 255.0

    print(len(flat_img))
    print('Func read_picture was successfully done')
    return flat_img.tolist()
# read_picture('one_plane/194_set1.jpg')

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
    Шукаємо egenvectors

    Результат -- матриця проектованих на 50 найважливіших векторів
    """
    # transposed_meaned_list_of_arrays = meaned_list_of_arrays.T
    
    A = Matrix(meaned_list_of_arrays)
    # A_T = [[A[j][i] for j in range(len(A))] for i in range(len(A[0]))]
    A_T = A.transpose()
    V = A * A_T / len(meaned_list_of_arrays)
    λ = symbols('λ')
    I = eye(V.shape[0])
    print('One more')

    # шукаємо детермінант, щоб прирівняти до 0
    determinant = (V - λ*I).det()
    eigvals = solve(determinant, λ)
    eigvals = sorted(eigvals, reverse=True)[:50]

    # eigenvectors = []
    # for eigval in eigvals:
    #     V_1 = V - eigval*I
    #     rref_matrix, pivot_columns = V_1.rref()
    #     eigenvectors.append(pivot_columns)

    # V_k = np.array(eigenvectors)
    # Z = [V_k * X for X in meaned_list_of_arrays] #проекції на 50 найважливіших "осей"
    # print('Func egenvectors was successfully done')
    # return Z

    eigenvectors = []
    for eigval in eigvals:
        V_1 = V - eigval * I
        nullspace = V_1.nullspace()
        if nullspace:
            eigenvectors.append(nullspace[0])
            print('dfxghj')

    V_k = Matrix.hstack(*eigenvectors)
    Z = [V_k.T * Matrix(x) for x in meaned_list_of_arrays]
    Z_np = np.array([[float(val) for val in vec] for vec in Z])
    print('Func egenvectors was successfully done')
    return Z_np, np.array(V_k.tolist(), dtype=float)

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
        """
        n_samples, n_features = X.shape

        y_ = np.where(y <= 0, -1, 1)

        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y_[idx]))
                    self.b -= self.lr * y_[idx]
        print('Func fit was successfully done')

    def predict(self, X):
        """
        Предікт -- передаємо матрицю зображення
        """
        approx = np.dot(X, self.w) - self.b
        print('Func predict was successfully done')
        return np.sign(approx)


def load_all_images(folder):
    """
    стягуємо файли з папок
    """
    files = sorted([os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.png')])
    return [read_picture(p) for p in files]

X0 = load_all_images('no_plane_01')     # клас 0
X1 = load_all_images('one_plane_01')    # клас 1

X_all = X0 + X1
mean_vector = find_mean(X_all)
Z, V_k = egenvectors(X_all)

y = [0] * len(X0) + [1] * len(X1)

svm = SVM()
print('SVM created')
svm.fit(Z, y)

print('Almost end')
def classify_new_image(path, mean_vector, V_k, svm):
    """"""
    x = read_picture(path)
    x_centered = np.array(x) - np.array(mean_vector)
    x_proj = np.dot(V_k.T, x_centered)
    prediction = svm.predict(x_proj.reshape(1, -1))
    return prediction[0]


print("Клас:", classify_new_image('one_to_check/SUV_083.png', mean_vector, V_k, svm))  # Має бути 1
# print("Клас:", classify_new_image('no_plane/tile_0.png', mean_vector, V_k, svm))   # Має бути -1



