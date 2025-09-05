# detect_planes_in_folder.py
"""
Даємо папку, де є фотки і воно виводить на скількох фотках і як вони звуться, було знайдено літками
"""

import os
import numpy as np
from PIL import Image
from main import classify_new_image, read_picture, mean_vector, V_k, svm 

def detect_planes_in_folder(folder_path):
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    planes_found = []

    for img_name in image_files:
        img_path = os.path.join(folder_path, img_name)

        try:
            pred = classify_new_image(img_path, mean_vector, V_k, svm)
            if pred == 1:
                planes_found.append(img_name)
        except Exception as e:
            print(f"Помилка з файлом {img_name}: {e}")

    print("\nРЕЗУЛЬТАТИ:")
    if not planes_found:
        print("Літаків не знайдено.")
    else:
        print(f"Знайдено літаки на {len(planes_found)} зображеннях:")
        for name in planes_found:
            print(f" - {name}")


if __name__ == "__main__":
    folder = "plane_from_coords_07"
    detect_planes_in_folder(folder)
