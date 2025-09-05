"""
Цей код зберігає у папку no_plane зображення 250*250 без літаків (за координатами) у кількості 200 штук 
Ye s pfufkjv ajnrb gj rjjhlbyfnf[]

Тобто з координат у відповідну папку завантажуємо фотки з супутника, мб треба перевірити чи воно "сусідні" квадратики бере, чи нічо не опускає
"""

import contextily as cx
import matplotlib.pyplot as plt
from shapely.geometry import box
import geopandas as gpd
from pyproj import Transformer
import numpy as np
import os

def download_tile(lat, lon, output_path='a_tile.png', size_m=250, zoom=18):
    # Перетворення координат з WGS84 до Web Mercator
    transformer = Transformer.from_crs("epsg:4326", "epsg:3857", always_xy=True)
    x_center, y_center = transformer.transform(lon, lat)

    # Обчислення меж прямокутника розміром size_m x size_m
    half_size = size_m / 2
    x_min = x_center - half_size
    x_max = x_center + half_size
    y_min = y_center - half_size
    y_max = y_center + half_size

    # Створення геометрії прямокутника
    bbox = box(x_min, y_min, x_max, y_max)
    geo = gpd.GeoDataFrame({'geometry': [bbox]}, crs='EPSG:3857')

    # Побудова зображення
    ax = geo.plot(figsize=(5, 5), alpha=0)
    cx.add_basemap(
        ax,
        crs=geo.crs.to_string(),
        zoom=zoom,
        source=cx.providers.Esri.WorldImagery  # без написів
    )

    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()


def generate_grid(lat_start, lon_start, lat_end, lon_end, step_m=250):
    # Перетворення метрів у градуси
    lat_step = step_m / 111320
    lon_step = step_m / (40075000 * np.cos(np.radians((lat_start + lat_end) / 2)) / 360)

    lat_points = np.arange(lat_start, lat_end, lat_step)
    lon_points = np.arange(lon_start, lon_end, lon_step)

    return [(lat, lon) for lat in lat_points for lon in lon_points]



# coordinates = generate_grid(49.55, 25.55, 49.65, 25.65)

# # Створимо папку, якщо нема
# output_dir = "no_no_plane"
# os.makedirs(output_dir, exist_ok=True)

# # Обмеження: згенеруй тільки 200 зображень
# for idx, (lat, lon) in enumerate(coordinates[:200]):
#     output_file = os.path.join(output_dir, f"a_tile_{idx}.png")
#     print(f"Зберігаю a_tile_{idx}.png...")
#     download_tile(lat, lon, output_path=output_file)

# print("згенеровано 200 зображень.")


coordinates = generate_grid(55.50, 37.49, 55.52, 37.51)
# x: 47.2733628, y: 39.6405598
# 47.280289, 39.646656    47.280636, 39.656610
# 47.275664, 39.646211    47.276052, 39.656699
# 47.1146048, y: 39.7915008
# 47.0826722, y: 39.4020447
# 47.4936056, y: 39.9240097
# 55.0692591 / 37.4549674
# 55.4085925 / 37.9067608
# 55.5122355 / 37.5070701
output_dir = "plane_from_coords_07"
os.makedirs(output_dir, exist_ok=True)

for idx, (lat, lon) in enumerate(coordinates[:100]):
    output_file = os.path.join(output_dir, f"a_tile_{idx}.png")
    print(f"Зберігаю a_tile_{idx}.png...")
    download_tile(lat, lon, output_path=output_file)

print("згенеровано 100 зображень")