from typing import List, TypeVar
import os

import cv2
import numpy as np
from numpy import ndarray
from tqdm import tqdm
from glob import glob
from pandas import DataFrame

from Data_Extrator import create_df, read_csv, concat_data, normalize_df
from Features_extractor import draw_contours, find_contours, draw_circle, find_moments, find_symmetry_x, \
    find_symmetry_y, find_variance_bgr, find_mean_bgr, find_binary_pixels, find_center_of_mass, \
    find_min_enclosing_circle, show_single_img

H: int = 256
W: int = 256

T = TypeVar("T")


def load_data_prod(path_to_dataset: str, images_folder_name: str, images_file_type: str) -> List[T]:
    list_images: List[T] = sorted(glob(os.path.join(path_to_dataset, images_folder_name, images_file_type)))
    return list_images


def read_image(path: str) -> ndarray:
    return cv2.imread(path, cv2.IMREAD_UNCHANGED)


def read_binary_image(path: str) -> ndarray:
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)


def resize(image: ndarray, mask: ndarray = None) -> ndarray:
    return cv2.resize(image, (H, W), interpolation=cv2.INTER_AREA)


def dilation(binary_img, iterations) -> ndarray:
    return cv2.dilate(binary_img, None, iterations=iterations)


dataset_binary: str = r"C:\Users\Raphael Nascimento\OneDrive\Documentos\Pesquisa"
binary_images: List[T] = load_data_prod(dataset_binary, "ISIC-2017_Training_Part1_GroundTruth", "*.png")

broken_images: list[str] = []

descriptors_df: DataFrame = create_df(
    columns=["image_name", "symmetry_x", "symmetry_y", "diameter", "mean_red", "mean_green", "mean_blue", "variance_red",
             "variance_green", "variance_blue"])

for actual_image in tqdm(binary_images, total=len(binary_images)):
    name: str = actual_image.split('\\')[-1]
    new_name: str = "_".join(name.split("_")[:2])
    new_name = f"{new_name}.jpg"

    bgr_image: ndarray = resize(
        read_image(rf"C:\Users\Raphael Nascimento\OneDrive\Documentos\Pesquisa\ISIC-2017_Training_Data\{new_name}"))

    binary_image: ndarray = resize(read_binary_image(actual_image))
    binary_image = dilation(binary_image, 1)
    # show_single_img(image, "a")

    contours, hierarchy = find_contours(binary_image)

    try:
        max_contours: ndarray = contours[0]
    except IndexError:
        print(f"\nThe image {new_name} is broken. Skipping.")
        broken_images.append(new_name.split('.')[0])
        continue

    for i in contours:
        if len(i) > len(max_contours):
            max_contours = i

    contours = max_contours

    moments = find_moments(binary_image)
    (cX, cY) = find_center_of_mass(moments)
    (rows, cols) = binary_image.shape
    binary_pixels = find_binary_pixels(binary_image, rows, cols)
    symmetry_x = find_symmetry_x(binary_image, rows, cols, cX)
    symmetry_y = find_symmetry_y(binary_image, rows, cols, cY)
    (x, y), diameter = find_min_enclosing_circle(contours)
    blue_mean, green_mean, red_mean = find_mean_bgr(bgr_image, binary_image,
                                                    rows,
                                                    cols,
                                                    binary_pixels)
    (blue_variance, green_variance, red_variance) = find_variance_bgr(
        bgr_image, binary_image, rows, cols, binary_pixels, blue_mean,
        green_mean, red_mean)

    descriptors: dict = {
        "image_name": new_name.split('.')[0],
        "symmetry_x": symmetry_x,
        "symmetry_y": symmetry_y,
        "diameter": diameter,
        "mean_red": red_mean,
        "mean_green": green_mean,
        "mean_blue": blue_mean,
        "variance_red": red_variance,
        "variance_green": green_variance,
        "variance_blue": blue_variance
    }

    new_df: DataFrame = create_df(data=[descriptors])

    descriptors_df = concat_data(descriptors_df, new_df)

    # bgr_image = draw_contours(bgr_image, contours, -1)
    # bgr_image = draw_circle(bgr_image, (int(cX), int(cY)), 3)
    # bgr_image = draw_circle(bgr_image, (int(x), int(y)), int(diameter))
    # show_single_img(bgr_image, "Result")


df_csv: DataFrame = read_csv("Data/ISIC-2017_Training_Part3_GroundTruth.csv")
df_csv = df_csv.drop(columns="seborrheic_keratosis", axis=1)

for broken in broken_images:
    df_csv = df_csv.drop(df_csv.loc[df_csv['image_id'] == broken].index)

df_csv = df_csv.drop(columns="image_id", axis=1)

descriptors_df.insert(0, "diagnose", df_csv.values)

descriptors_df.to_csv(path_or_buf="Data/golden-descriptors.csv", index=False)

descriptors_df = descriptors_df.drop(columns="image_name", axis=1)

descriptors_df = normalize_df(descriptors_df)
descriptors_df.to_csv(path_or_buf="Data/golden-normalized-descriptors.csv", index=False)

print("Done")
