import numpy as np
import cv2
from scipy.ndimage.measurements import label
from data import load_database
from feature_extraction import convert_color, bin_spatial, color_histogram, get_hog_features

__db = load_database()


def draw_boxes(img, bounding_boxes, color=(0, 0, 255), thickness=6):
    """
    Takes a list of bounding boxes (denoted by the coordinates of their lower left corner and their upper right corner)
    and draws them on the input image.
    """
    image_with_boxes = np.copy(img)

    # draw each bounding box on your image copy using cv2.rectangle()
    for corner_1, corner_2 in bounding_boxes:
        cv2.rectangle(image_with_boxes, corner_1, corner_2, color, thickness)

    return image_with_boxes


def find_cars(img, scales, classifier, scaler, parameters, x_start, x_stop, y_start, y_stop):
    """
    Identifies the cars in a picture.
    :return: Same input image with cars identified inside bounding boxes.
    """
    def find_at_scale(region_boundaries, scale):
        """
        Finds cars in the input image after resizing to a particular scale.
        """
        x_start, y_start, x_stop, y_stop = region_boundaries
        image_region = img[y_start:y_stop, x_start:x_stop, :]
        color_transformed_region = convert_color(image_region, parameters['color_space'])

        if scale != 1:
            region_shape = color_transformed_region.shape
            new_shape = (np.int(region_shape[1] / scale), np.int(region_shape[0] / scale))
            color_transformed_region = cv2.resize(color_transformed_region, new_shape)

        # Unpack channels
        channel_1 = color_transformed_region[:, :, 0]
        channel_2 = color_transformed_region[:, :, 1]
        channel_3 = color_transformed_region[:, :, 2]

        # Dimensions
        width, height = channel_1.shape[1], channel_1.shape[0]

        # Define blocks and steps
        number_of_blocks_in_x = (width // parameters['pix_per_cell']) - 1
        number_of_blocks_in_y = (height // parameters['pix_per_cell']) - 1

        # 64 was the original sampling rate, with 8 cells and 8 pix per cell
        window = 64
        number_of_blocks_per_window = (window // parameters['pix_per_cell']) - 1
        cells_per_step = 2  # Instead of overlap, define how many cells to step
        number_of_steps_in_x = (number_of_blocks_in_x - number_of_blocks_per_window) // cells_per_step
        number_of_steps_in_y = (number_of_blocks_in_y - number_of_blocks_per_window) // cells_per_step

        # Compute individual channel HOG features for the entire region
        all_channels_hogs = [
            get_hog_features(channel_1, orient=parameters['orientations'], pix_per_cell=parameters['pix_per_cell'],
                             cell_per_block=parameters['cell_per_block'], feature_vector=False),
            get_hog_features(channel_2, orient=parameters['orientations'], pix_per_cell=parameters['pix_per_cell'],
                             cell_per_block=parameters['cell_per_block'], feature_vector=False),
            get_hog_features(channel_3, orient=parameters['orientations'], pix_per_cell=parameters['pix_per_cell'],
                             cell_per_block=parameters['cell_per_block'], feature_vector=False)
        ]

        car_windows = []
        for xb in range(number_of_steps_in_x):
            for yb in range(number_of_steps_in_y):
                ypos = yb * cells_per_step
                xpos = xb * cells_per_step

                # Extract HOG for this patch
                if parameters['hog_channels'] == 'ALL':
                    hogs_considered = [hog_feat[ypos:ypos + number_of_blocks_per_window,
                                       xpos:xpos + number_of_blocks_per_window].ravel() for hog_feat in
                                       all_channels_hogs]
                else:
                    hogs_considered = [all_channels_hogs[channel][ypos:ypos + number_of_blocks_per_window,
                                       xpos:xpos + number_of_blocks_per_window].ravel() for channel in
                                       parameters['hog_channels']]

                hog_features = np.hstack(hogs_considered)

                xleft = xpos * parameters['pix_per_cell']
                ytop = ypos * parameters['pix_per_cell']

                # Extract the image patch
                image_patch = cv2.resize(color_transformed_region[ytop:ytop + window, xleft:xleft + window], (64, 64))

                features = [hog_features]
                # Get color features

                if parameters['histogram_features']:
                    hist_features = color_histogram(image_patch, number_of_bins=parameters['number_of_bins'])
                    features.insert(0, hist_features)

                if parameters['spatial_features']:
                    spatial_features = bin_spatial(image_patch, size=parameters['spatial_size'])
                    features.insert(0, spatial_features)

                # Scale features and make a prediction
                features = np.hstack(features).reshape(1, -1)

                test_features = scaler.transform(features)
                test_prediction = classifier.predict(test_features)

                if test_prediction == 1:
                    xbox_left = np.int(xleft * scale)
                    ytop_draw = np.int(ytop * scale)
                    win_draw = np.int(window * scale)
                    new_window = ((xbox_left + x_start, ytop_draw + y_start),
                                  (xbox_left + x_start + win_draw, ytop_draw + win_draw + y_start))
                    car_windows.append(new_window)

        return car_windows

    if not y_start:
        y_start = 0

    if not x_start:
        x_start = 0

    if not y_stop:
        y_stop = img.shape[0]

    if not x_stop:
        x_stop = img.shape[1]

    car_windows = []

    region_boundaries = (x_start, y_start, x_stop, y_stop)

    for scale in scales:
        car_windows += find_at_scale(region_boundaries, scale)

    return car_windows


def get_heatmap_canvas(image):
    """
    Generates an empty heatmap canvas shaped as the input image.
    :param image: Input image.
    :return: Empty canvas filled with zeros ant the same shape as the input image.
    """
    return np.zeros_like(image[:, :, 0]).astype(np.float64)


def add_heat(heatmap, bounding_boxes_list):
    """
    Takes a list of bounding boxes and "increases the heat" of the pixels bounded by them.
    :param heatmap: Heatmap to be altered.
    :param bounding_boxes_list: List of rectangular areas where the heat will be increased,
    :return: Copy of the heatmat with the heat increased.
    """
    # Iterate through list of bounding boxes
    for box in bounding_boxes_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    return heatmap


def apply_threshold(heatmap, threshold):
    """
    Zeroes out the pixels below the heat threshold.
    :param heatmap: Heatmap to be operated on.
    :param threshold: Minimum heat required.
    :return: Thresholded heatmap.
    """
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap


def draw_labeled_bounding_boxes(image, labeled_heatmap, number_of_cars):
    """
    Draws the bounding boxes corresponding to the car detections in the input image.
    """
    # Iterate through all detected cars
    for car_number in range(1, number_of_cars + 1):
        # Find pixels with each car_number label value
        nonzero = (labeled_heatmap == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzero_y = np.array(nonzero[0])
        nonzero_x = np.array(nonzero[1])

        # Define a bounding box based on min/max x and y
        upper_left_corner = (np.min(nonzero_x), np.min(nonzero_y))
        lower_right_corner = (np.max(nonzero_x), np.max(nonzero_y))

        # Draw the box on the image
        cv2.rectangle(image, upper_left_corner, lower_right_corner, (0, 0, 255), 6)

    return image


def get_labeled_cars(heatmap):
    """
    Takes a heatmap and returns the labels of the hottest regions (which represent cars) and the number of cars
    identified
    :param heatmap:
    :return: A heatmap image where each labeled region will be tagged with a particular number. Also, the number of
    cars found.
    """
    labeled_heatmap, number_of_cars = label(heatmap)
    return labeled_heatmap, number_of_cars


def process_image(img, scales, classifier, scaler, parameters, x_start=None, x_stop=None, y_start=None, y_stop=None,
                  heatmaps=[], heatmap_threshold=4):
    """
    Processes an image, returning a copy with cars identified and enclosed by bounding boxes. It smooths the detections
    by taking into account the heatmaps of the last N frames indicated by the value of the 'smoothing_factor' key in
    the 'parameters' dictionary.
    """
    # Extract windows.
    car_windows = find_cars(img=img, x_start=x_start, x_stop=x_stop, y_start=y_start, y_stop=y_stop,
                            parameters=parameters, scales=scales, classifier=classifier, scaler=scaler)

    # Compute the heatmap for this frame and add heat.
    heatmap = get_heatmap_canvas(img)
    heatmap = add_heat(heatmap, car_windows)

    heatmaps.append(heatmap)

    # Compute the mean of the last 'smoothing_factor' frames' heatmaps and threshold it.
    heatmaps_mean = np.mean(heatmaps[-parameters['smoothing_factor']:], axis=0)
    heatmaps_mean = apply_threshold(heatmaps_mean, heatmap_threshold)

    # Merge several windows by computing the labels of the cars identified.
    labeled_heatmap, number_of_cars = get_labeled_cars(heatmaps_mean)

    # Finally, draw results on the input image.
    draw_image = draw_labeled_bounding_boxes(img, labeled_heatmap, number_of_cars)

    return draw_image

