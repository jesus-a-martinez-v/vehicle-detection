"""
Feature extraction helper functions.
"""

import numpy as np
import cv2
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog


def get_normalizer(data):
    """
    Takes a dataset and fits a normalizer to it.
    :param data: Dataset used to fit the normalizer.
    :return: Returns a fit instance of StandardScaler.
    """
    scaler = StandardScaler().fit(data)
    return scaler


def color_histogram(image, number_of_bins=32):
    """
    Calculates the color histogram features of an image.
    :param image: Image used to compute the histograms.
    :param number_of_bins: Number of bins for each histogram
    :return: A vector that's the concatenation of the computed histograms on each of the color channels.
    """
    # Compute the histogram of the color channels separately
    first_channel = image[:, :, 0]
    second_channel = image[:, :, 1]
    third_channel = image[:, :, 2]

    first_channel_histogram = np.histogram(first_channel, bins=number_of_bins)
    second_channel_histogram = np.histogram(second_channel, bins=number_of_bins)
    third_channel_histogram = np.histogram(third_channel, bins=number_of_bins)

    # Concatenate the histograms into a single feature vector
    histogram_features = np.concatenate((first_channel_histogram[0], second_channel_histogram[0],
                                         third_channel_histogram[0]))
    # Return the individual histograms, bin_centers and feature vector
    return histogram_features


def get_hog_features(img, orient, pix_per_cell, cell_per_block, visualise=False, feature_vector=True):
    """
    Computes the Histogram of Oriented Gradients of an input image.
    :param img: Input image.
    :param orient: Number of orientations (bins) to use.
    :param pix_per_cell: Number of pixes that comprise a cell.
    :param cell_per_block: Number of cells that compose a block.
    :param visualise: Flag that indicates if we should return a visualization image of the HOGs calculated.
    :param feature_vector: Flag that indicates if we should return the result as a 1-D array.
    :return: A tuple comprised of a visualization and the actual features if visualise=True, or just the features otherwise.
    """
    hog_result = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                     cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True, visualise=visualise,
                     feature_vector=feature_vector)

    if visualise:
        return hog_result[0], hog_result[1]
    else:
        return hog_result


def convert_color(image, color_space='RGB'):
    """
    Converts an image (assumed to be in BGR) to a desired color space.
    :param image: Input image.
    :param color_space: Target color space.
    :return: Copy of the image converted to the target color space.
    """
    color_space = color_space.lower()
    if color_space != 'rgb':
        if color_space == 'hsv':
            color_transformation = cv2.COLOR_BGR2HSV
        elif color_space == 'luv':
            color_transformation = cv2.COLOR_BGR2LUV
        elif color_space == 'hls':
            color_transformation = cv2.COLOR_BGR2HLS
        elif color_space == 'yuv':
            color_transformation = cv2.COLOR_BGR2YUV
        elif color_space == 'ycrcb':
            color_transformation = cv2.COLOR_BGR2YCrCb
        else:
            raise ValueError('Invalid value %s for color_space parameters. Valid color spaces are: RGB, HSV, LUV, '
                             'HLS, YUV, YCrCb' % color_space)

        return cv2.cvtColor(image, color_transformation)
    else:
        return image


def bin_spatial(image, size=(32, 32)):
    """
    Takes an image, resizes it and converts it into a vector.
    :param image: Image to be converted.
    :param size: Target size (width, height)
    :return: Feature vector that's the result of flattening the original image after resizing it to the target size.
    """
    color1 = cv2.resize(image[:, :, 0], size).ravel()
    color2 = cv2.resize(image[:, :, 1], size).ravel()
    color3 = cv2.resize(image[:, :, 2], size).ravel()
    return np.hstack((color1, color2, color3))


def single_image_features(image, color_space='RGB', spatial_size=(32, 32), histogram_bins=32, orient=9, pix_per_cell=8,
                          cell_per_block=2, hog_channels='ALL', spatial_feat=True, hist_feat=True, hog_feat=True):
    """
    Extract features from a single image.
    :return: A feature vector that is the concatenation of the computed individual features marked as True
    """
    # 1) Define an empty list to receive features
    image_features = []

    # 2) Apply color conversion if other than 'RGB'
    feature_image = convert_color(image, color_space)

    # 3) Compute spatial features if flag is set
    if spatial_feat:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        # 4) Append features to list
        image_features.append(spatial_features)

    # 5) Compute histogram features if flag is set
    if hist_feat:
        hist_features = color_histogram(feature_image, number_of_bins=histogram_bins)
        # 6) Append features to list
        image_features.append(hist_features)

    # 7) Compute HOG features if flag is set
    if hog_feat:
        if hog_channels == 'ALL':
            channels_considered = range(feature_image.shape[-1])
        else:
            channels_considered = hog_channels

        hog_features = []

        for channel in channels_considered:
            hog_feats_for_channel = get_hog_features(img=feature_image[:, :, channel], orient=orient,
                                                     pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                                                     visualise=False, feature_vector=True)
            hog_features.extend(hog_feats_for_channel)

        # 8) Append features to list
        image_features.append(hog_features)

    # 9) Return concatenated array of features
    return np.concatenate(image_features)


def extract_features(images, color_space='RGB', spatial_size=(32, 32), histogram_bins=32, orient=9, pix_per_cell=8,
                     cell_per_block=2, hog_channels='ALL', spatial_feat=True, hist_feat=True, hog_feat=True):
    """
    Extract features of a whole set of images.
    :return: A feature vector that is the concatenation of the computed individual features marked as True for each image.
    """
    # Create a list to append feature vectors to
    images_feature_vectors = []
    # Iterate through the list of images
    for img in images:
        feature_vector = single_image_features(img, color_space=color_space, spatial_size=spatial_size,
                                               histogram_bins=histogram_bins, orient=orient, pix_per_cell=pix_per_cell,
                                               cell_per_block=cell_per_block, hog_channels=hog_channels,
                                               spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat)
        images_feature_vectors.append(feature_vector)

    # Return list of feature vectors
    return images_feature_vectors


if __name__ == '__main__':
    # # Generate a random index to look at a car image
    # ind = np.random.randint(0, len(cars))
    # Read in the image
    from data import load_database
    import matplotlib.image as mpimg

    db = load_database()
    cars = db['cars_features']
    non_cars = db['non_cars_features']

    # Test folder
    path = "./output_images/hog/"
    # Image
    car_image = cars[int(np.random.uniform(0, len(cars)))]
    non_car_image = non_cars[int(np.random.uniform(0, len(non_cars)))]

    # Param grid

    for color_space in ['hsv', 'ycrcb']:
        for pix_per_cell in [4, 8, 16]:
            for orientation in [8, 9, 24, 32]:
                for cell_per_block in [1, 2]:
                    converted_car_image = convert_color(car_image, color_space)
                    converted_non_car_image = convert_color(non_car_image, color_space)

                    for channel in [0, 1, 2]:
                        _, car_visualization = get_hog_features(img=converted_car_image[:, :, channel],
                                                                pix_per_cell=pix_per_cell,
                                                                orient=orientation,
                                                                cell_per_block=cell_per_block,
                                                                visualise=True)
                        _, non_car_visualization = get_hog_features(img=converted_non_car_image[:, :, channel],
                                                                    pix_per_cell=pix_per_cell,
                                                                    orient=orientation,
                                                                    cell_per_block=cell_per_block,
                                                                    visualise=True)

                        # Save them
                        mpimg.imsave(path + "car_original.jpg", car_image)
                        mpimg.imsave(path + "car_" + color_space + ".jpg", converted_car_image)
                        mpimg.imsave(path + "car_colorspace=" + color_space + "_channel=" + str(channel) + ".jpg",
                                     converted_car_image[:, :, channel])
                        mpimg.imsave(path + "car_hog_colorspace=" + color_space + "_channel=" + str(channel) +
                                     "_pix_per_cell=" + str(pix_per_cell) + "_orientation=" + str(orientation) +
                                     "_cell_per_block=" + str(cell_per_block) + ".jpg", car_visualization)

                        mpimg.imsave(path + "non_car_original.jpg", non_car_image)
                        mpimg.imsave(path + "non_car_" + color_space + ".jpg", converted_non_car_image)
                        mpimg.imsave(path + "non_car_colorspace=" + color_space + "_channel=" + str(channel) + ".jpg",
                                     converted_non_car_image[:, :, channel])
                        mpimg.imsave(path + "non_car_hog_colorspace=" + color_space + "_channel=" + str(channel) +
                                     "_pix_per_cell=" + str(pix_per_cell) + "_orientation=" + str(orientation) +
                                     "_cell_per_block=" + str(cell_per_block) + ".jpg", non_car_visualization)
