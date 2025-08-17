import cv2
import numpy as np
import math

from scipy.optimize import linear_sum_assignment

"""
    Line follower
"""


def four_point_transform(image, rect, vert_size):
    # obtain a consistent order of the points and unpack them
    # individually
    (tl, tr, br, bl) = rect

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth-1, 0],
        [maxWidth-1, maxHeight-1],
        [0, maxHeight-1]], dtype="float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # return the warped image
    return warped, maxHeight

def get_roi(img,
            vert_cutting_factor,
            corner_up_factor,
            corner_down_factor):
    vert_size = int(img.shape[0] * vert_cutting_factor)

    # cut image corners
    polygon_up = int(img.shape[1] * corner_up_factor)
    polygon_down = int(img.shape[1] * corner_down_factor)

    tri_img = img.copy()

    rect = np.array([
            [polygon_up, img.shape[0]-vert_size],
            [img.shape[1]-polygon_up, img.shape[0]-vert_size],
            [img.shape[1]-polygon_down, img.shape[0]-1],
            [polygon_down, img.shape[0]-1]], dtype="float32")

    img_mod, vert_size = four_point_transform(tri_img, rect, vert_size)

    return img_mod, img.shape[0] - vert_size


def thresh_and_process(hsv_img, low_thresh, up_thresh):
    thresh = cv2.inRange(hsv_img, low_thresh, up_thresh)
    mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    return mask


def find_biggest_contour(mask, out="mask"):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    max_area = 0
    global largest_contour
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            largest_contour = contour

    if out == "mask":
        output = np.zeros_like(mask)
        cv2.drawContours(output, [largest_contour], -1, 255, thickness=cv2.FILLED)
        return output
    elif out == "cont":
        return largest_contour


def cut_image_into_portions(img, n_por=5):
    step = math.floor(img.shape[0] / n_por)

    for i_v in range(n_por):
        cv2.line(img, [0, i_v * step], [img.shape[1], i_v * step], 0, thickness=1)


def get_contour_centroids(img):
    centroids = []

    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        M = cv2.moments(contour)
        if M["m00"] != 0:  # Avoid division by zero
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.circle(img, (cx, cy), 3, 0, -1)
            centroids.append([cx, cy])

    return centroids


def sort_normally(centroids1, centroids2):
    array1 = np.array(centroids1)
    array2 = np.array(centroids2)

    sorted_white = array1[np.argsort(array1[:, 1])]
    sorted_yellow = array2[np.argsort(array2[:, 1])]

    return sorted_white, sorted_yellow


def sort_by_distance(centroids1, centroids2):

    array1 = np.array(centroids1)
    array2 = np.array(centroids2)

    y1 = array1[:, 1]
    y2 = array2[:, 1]

    cost_matrix = np.abs(y1[:, np.newaxis] - y2)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    assigned_points = array2[col_ind] # array2 sorted to minimize distance to array1

    return assigned_points, array1


def normalize_and_agregate(shape, l_points, r_points):
    center_coords = []
    for i in range(l_points.shape[0]):
        center_coords.append([round(((l_points[i][0] + r_points[i][0]) / 2) - shape[1] / 2), round(((l_points[i][1] + r_points[i][1]) / 2) - shape[1] / 2)])

    # normalize x-axis
    for point in center_coords:
        point[0] = round(point[0] / (shape[1] / 2), 3)

    # normalize y-axis logarithmically
    f = lambda x : 0.2 * math.log(x) + 0.7

    upper_limit = math.e ** (3/2)
    lower_limit = math.e ** -(7/2)

    diff = upper_limit - lower_limit
    step = diff / len(center_coords)

    fin_sum = 0
    for i, point in enumerate(center_coords):
        point[1] = round(f(upper_limit - i * step), 3)
        fin_sum += point[0] * point[1]

    return round(fin_sum / len(center_coords), 3)


def process(frame):
    img, vert_split = get_roi(frame, 0.6, 0.33, 0.0) # getting appropriate ROI
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) # converting to HSV scale

    mask_yellow = thresh_and_process(hsv, (8, 154, 130), (180, 255, 255))
    mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    mask_white = thresh_and_process(hsv, (0, 0, 129), (180, 255, 255))
    mask_white = cv2.morphologyEx(mask_white, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    mask_white = cv2.morphologyEx(mask_white, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    only_yellow = cv2.bitwise_and(mask_yellow, mask_white)
    only_white = cv2.subtract(mask_white, only_yellow)

    return only_yellow, only_white


def process_lines(only_yellow, only_white):
    white_big = find_biggest_contour(only_white)
    yellow_big = find_biggest_contour(only_yellow)

    cut_image_into_portions(white_big)
    cut_image_into_portions(yellow_big)

    white_centroids = get_contour_centroids(white_big)
    yellow_centroids = get_contour_centroids(yellow_big)

    if len(white_centroids) > len(yellow_centroids): left_points, right_points = sort_by_distance(yellow_centroids, white_centroids)
    elif len(white_centroids) < len(yellow_centroids): right_points, left_points = sort_by_distance(white_centroids, yellow_centroids)
    else: right_points, left_points = sort_normally(white_centroids, yellow_centroids)

    final_error = normalize_and_agregate(only_yellow.shape, left_points, right_points)
    return final_error


"""
    Intersection detection
"""


def approx_intersection_contour(mask_yellow):
    yellow_cont = find_biggest_contour(mask_yellow)
    epsilon = 0.01 * cv2.arcLength(yellow_cont, True) # type: ignore
    approx = cv2.approxPolyDP(yellow_cont, epsilon, True) # type: ignore
    return approx


def fit_and_check_orientation(approximation_contour):
    [vx, vy, x, y] = cv2.fitLine(approximation_contour, cv2.DIST_L2, 0, 0.01, 0.01)
    angle = np.arctan2(vy, vx) * 180 / np.pi
    is_horizontal = bool(abs(angle) < 10 or abs(angle) > 170)
    return is_horizontal


def is_intersection(only_yellow):
    approx = approx_intersection_contour(only_yellow)
    return fit_and_check_orientation(approx)