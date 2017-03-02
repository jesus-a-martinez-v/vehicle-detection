"""
Loads an prepares the data.
"""

import cv2
import numpy as np
import glob
import pickle
import os


def __load_images_from_directory(path, dataset_type):
    """

    :param path:
    :param dataset_type:
    :return:
    """
    data = []

    for image_file in glob.glob(path + '/*/*.png'):
        image = cv2.imread(image_file)
        data.append(image)

    if dataset_type == 'car':
        labels = np.ones(len(data))
    elif dataset_type == 'non car':
        labels = np.zeros(len(data))
    else:
        raise ValueError('Unexpected value %s for "dataset_type" param. Valid values: "car", "non car"')

    return data, labels


def load_database(path='./db.p'):
    """

    :param path:
    :return:
    """
    if os.path.exists(path):
        with open(path, 'rb') as pickle_file:
            db = pickle.load(pickle_file)
    else:
        with open(path, 'wb') as pickle_file:
            db = {}
            pickle.dump(db, pickle_file)

    return db


def load_cars(path='./vehicles/vehicles'):
    """
    Loads all cars images.
    :param path: Directory path where the car images are.
    :return: Array of images and their corresponding labels (1)
    """

    return __load_images_from_directory(path, 'car')


def load_non_cars(path='./non-vehicles/non-vehicles'):
    """
    Loads all non car images.
    :param path: Directory path where the non car images are.
    :return: Array of images and their corresponding labels (0).
    """

    return __load_images_from_directory(path, 'non car')


def load_data(save_at='db.p'):
    """

    :param save_at:
    :return:
    """
    cars_data, cars_labels = load_cars()
    non_cars_data, non_cars_labels = load_non_cars()

    print("Number of CARS examples:", len(cars_labels))
    print("Number of NON CARS examples:", len(non_cars_labels))

    if save_at:

        db = load_database(save_at)

        with open(save_at, 'wb') as pickle_file:
            print("About to save car and non car data at", save_at)

            db['cars_features'] = cars_data
            db['cars_labels'] = cars_labels
            db['non_cars_features'] = non_cars_data
            db['non_cars_labels'] = non_cars_labels

            pickle.dump(db, pickle_file)
            print("Saved! :)")

    return cars_data, cars_labels, non_cars_data, non_cars_labels


def save_parameters(save_at='db.p', params=None):
    if not params:
        params = {}  # Just to prevent NoneType related errors. Below we're setting some default values.

    db_parameters = {
        'number_of_bins': params.get('number_of_bins', 32),
        'color_space': params.get('color_space', 'YCrCb'),
        'spatial_size': params.get('spatial_size', (32, 32)),
        'orientations': params.get('orientations', 24),
        'pix_per_cell': params.get('pix_per_cell', 8),
        'cell_per_block': params.get('cell_per_block', 2),
        'hog_channels': params.get('hog_channels', 'ALL'),
        'spatial_features': params.get('spatial_features', False),
        'hog_features': params.get('hog_features', True),
        'scales': params.get('scales', (0.75, 1.5, 2, 2.25)),
        'histogram_features': params.get('histogram_features', False),
        'smoothing_factor': params.get('smoothing_factor', 5)
    }

    if save_at:

        db = load_database(save_at)

        with open(save_at, 'wb') as pickle_file:
            print("About to save feature extraction parameters at", save_at)
            db['parameters'] = db_parameters
            pickle.dump(db, pickle_file)
            print("Saved!")

    return db_parameters


if __name__ == '__main__':
    car_images, _, non_car_images, _ = load_data()
    save_parameters()
