"""
Цей файл для сортування фото з бази даних і вираховуванння тільки тих, де 1 літак по опису фотки з тієї бази
"""
import glob
import shutil
import os

destination_dir = "./one_plane"

files = glob.glob("D:/Старе завантаження/train/*.txt")
print(files)
one_plane_file = []
for file in files:
    with open(file, 'r', encoding='UTF-8') as file_opened:
            # print('fghj')
        lines = file_opened.read().split('\n')
        n_planes = len(lines)
        if n_planes == 1:
            one_plane_file.append(file)

print(one_plane_file)

for src_path in one_plane_file:
    jpg_path = os.path.splitext(src_path)[0] + '.jpg'

    filename = os.path.basename(jpg_path)  # тільки назва файлу
    dst_path = os.path.join(destination_dir, filename)
    try:
        shutil.copy(jpg_path, dst_path)
        print(f"Copied: {jpg_path} -> {dst_path}")
    except Exception as e:
        print(f"Got error {jpg_path}: {e}")

