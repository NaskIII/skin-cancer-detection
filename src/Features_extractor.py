import numpy as np
import cv2
from numpy import ndarray


def find_contours(image: ndarray):
    return cv2.findContours(image, cv2.RETR_TREE,
                            cv2.CHAIN_APPROX_NONE)


def find_moments(contours):
    return cv2.moments(contours)


def find_center_of_mass(moments):
    cx = int(moments["m10"] / moments["m00"])
    cy = int(moments["m01"] / moments["m00"])
    return cx, cy


def find_binary_pixels(binary_img, rows, cols):
    binary_pixels = 0
    for i in range(rows):
        for j in range(cols):
            if binary_img[i, j] == 255:
                binary_pixels += 1
    return binary_pixels


def find_symmetry_x(binary_img, rows, cols, cx):
    x_1 = x_2 = 0
    for i in range(rows):
        if i < cx:
            for j in range(cols):
                x_1 += binary_img[i, j]
        else:
            for j in range(cols):
                x_2 += binary_img[i, j]
    return abs(x_1 - x_2)


def find_symmetry_y(binary_img, rows, cols, cy):
    y_1 = y_2 = 0
    for i in range(rows):
        for j in range(cols):
            if j < cy:
                y_1 += binary_img[i, j]
            else:
                y_2 += binary_img[i, j]
    return abs(y_1 - y_2)


def find_min_enclosing_circle(coordinates):
    return cv2.minEnclosingCircle(coordinates)


def find_mean_bgr(image_bgr, binary_img, rows, cols, binary_pixels):
    blue_mean = green_mean = red_mean = 0
    for i in range(rows):
        for j in range(cols):
            (b, g, r) = image_bgr[i, j]
            blue_mean += int(b) * int(binary_img[i, j])
            green_mean += int(g) * int(binary_img[i, j])
            red_mean += int(r) * int(binary_img[i, j])
    blue_mean = (1 / binary_pixels) * blue_mean
    green_mean = (1 / binary_pixels) * green_mean
    red_mean = (1 / binary_pixels) * red_mean

    return blue_mean, green_mean, red_mean


def find_variance_bgr(image_bgr, binary_img, rows, cols, binary_pixels,
                      blue_mean, green_mean, red_mean):
    blue_variance = green_variance = red_variance = 0
    for i in range(rows):
        for j in range(cols):
            (b, g, r) = image_bgr[i, j]
            blue_variance += (int(b) - blue_mean) ** 2 * int(binary_img[i, j])
            green_variance += (int(g) - green_mean) ** 2 * int(binary_img[i, j])
            red_variance += (int(r) - red_mean) ** 2 * int(binary_img[i, j])
    blue_variance = (1 / binary_pixels) * blue_variance
    green_variance = (1 / binary_pixels) * green_variance
    red_variance = (1 / binary_pixels) * red_variance

    return blue_variance, green_variance, red_variance


def show_single_img(img, bar_text):
    cv2.imshow(bar_text, img)
    cv2.waitKey(0)


def show2img(img1, img2, bar_text):
    img_screen = np.vstack([np.hstack([img1, img2])])
    cv2.imshow(bar_text, img_screen)
    cv2.waitKey(0)


def show4img(img1, img2, img3, img4, bar_text):
    img_screen = np.vstack([np.hstack([img1, img2]),
                           np.hstack([img3, img4])])
    cv2.imshow(bar_text, img_screen)
    cv2.waitKey(0)


def draw_circle(img, coordinates, radius):
    return cv2.circle(img, coordinates, int(radius), (0, 255, 0), 3)


def draw_contours(img, coordinates, index):
    return cv2.drawContours(img, coordinates, index, (0, 255, 0), 3)


def save(name_img, img):
    cv2.imwrite(name_img, img)


class FeaturesExtractor(object):
    pass
