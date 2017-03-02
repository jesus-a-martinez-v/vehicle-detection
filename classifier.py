"""
Car/Not Car classifier
"""

from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, GridSearchCV
from feature_extraction import extract_features, get_normalizer
from data import load_database
import time
import pickle
import numpy as np

__db = load_database()


def prepare_data(save_scaler=True, location='./db.p'):
    """
    Prepares the data for training. It does so by extracting the features from both cars and non cars datasets.
    :param save_scaler: Flag that indicates if we should persist our scaler in database (i.e. the pickle file).
    :param location: Path of the pickle file that contains the data.
    :return: Features and labels ready to be passed to a classifier.
    """

    print("Loading training data")
    cars = __db['cars_features']
    cars_labels = __db['cars_labels']
    non_cars = __db['non_cars_features']
    non_cars_labels = __db['non_cars_labels']

    parameters = __db['parameters']

    print("Extracting features...")
    cars_extracted_features = extract_features(images=cars,
                                               color_space=parameters['color_space'],
                                               hog_channels=parameters['hog_channels'],
                                               orient=parameters['orientations'],
                                               pix_per_cell=parameters['pix_per_cell'],
                                               cell_per_block=parameters['cell_per_block'],
                                               histogram_bins=parameters['number_of_bins'],
                                               spatial_size=parameters['spatial_size'],
                                               spatial_feat=parameters['spatial_features'],
                                               hist_feat=parameters['histogram_features'],
                                               hog_feat=parameters['hog_features'])
    non_cars_extracted_features = extract_features(images=non_cars,
                                                   color_space=parameters['color_space'],
                                                   hog_channels=parameters['hog_channels'],
                                                   orient=parameters['orientations'],
                                                   pix_per_cell=parameters['pix_per_cell'],
                                                   cell_per_block=parameters['cell_per_block'],
                                                   histogram_bins=parameters['number_of_bins'],
                                                   spatial_size=parameters['spatial_size'],
                                                   spatial_feat=parameters['spatial_features'],
                                                   hist_feat=parameters['histogram_features'],
                                                   hog_feat=parameters['hog_features'])

    features = np.vstack((cars_extracted_features, non_cars_extracted_features)).astype(np.float64)
    labels = np.hstack((cars_labels, non_cars_labels))

    print("Normalizing...")
    scaler = get_normalizer(features)
    if save_scaler:
        with open(location, 'wb') as pickle_file:
            __db['scaler'] = scaler
            pickle.dump(__db, pickle_file)

    features = scaler.transform(features)

    print("Done!")
    return features, labels


def train_model(features, labels, test_proportion=0.2, seed=9991, save_model=True, location='./db.p'):
    """
    Takes a set of features and labels and trains a classifier on the car/non car dataset.
    :param save_model: Flag that indicates if we should persist our fit model in database (i.e. the pickle file).
    :param features: Dataset comprised of extracted features of the cars and non cars images.
    :param labels: Labels for the cars (1) and non cars (0) features vectors.
    :param test_proportion: Proportion of the dataset that'll be reserved for testing.
    :param seed: Seed for the randomization of the data during the splitting phase.
    :return: Fit model.
    """
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=test_proportion, random_state=seed)

    previous_training_parameters_exist = 'classifier_params' in __db

    # If there are previous training parameters, we assume they are the best, so we'll just use them. Otherwise,
    # perform a grid search for the best possible parameters.
    if previous_training_parameters_exist:
        clf_params = __db['classifier_params']
        print("Using these parameters for training:", clf_params)
        classifier = LinearSVC(C=clf_params['C'], loss=clf_params['loss'], max_iter=clf_params['max_iter'])
    else:
        param_grid = {
            'C': (1.0, 5.0, 0.5, 0.25),
            'loss': ('hinge', 'squared_hinge'),
            'max_iter': (500, 1000, 3000)
        }
        classifier = GridSearchCV(LinearSVC(), param_grid=param_grid, n_jobs=3, verbose=3)

    # Start training and measure time
    start = time.time()
    classifier.fit(X_train, y_train)
    end = time.time()

    # Print useful information
    print(round(end - start, 2), 'Seconds to train classifier...')
    # Check the score of the classifier
    print('Test Accuracy of the classifier = ', 100 * round(classifier.score(X_test, y_test), 4))

    # Save model and its parameters if the corresponding flag is on.
    if save_model:
        with open(location, 'wb') as pickle_file:
            if not previous_training_parameters_exist:
                __db['classifier_params'] = classifier.best_params_

            __db['model'] = classifier
            pickle.dump(__db, pickle_file)

    return classifier


if __name__ == '__main__':
    feats, lbls = prepare_data()

    clf = train_model(feats, lbls)

    print("Picking random example:")
    random_index = int(np.random.uniform(0, len(lbls)))
    print("Classifier prediction: ", clf.predict(feats[random_index]))
    print("Actual label: ", lbls[random_index])
    print("Classifier:", clf)
