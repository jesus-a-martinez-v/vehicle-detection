# Vehicle Detection
This project was developed as part of the Computer Vision module of the amazing Self-Driving Car Engineer Nanodegree
program offered by
 [![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)


The goals / steps of this project are the following:  

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a Linear SVM classifier
* [Optionally] Apply a color transform and append binned color features, as well as histograms of color, to the HOG feature vector. 
* Implement a sliding-window technique and use our trained classifier to search for vehicles in images.
* Run the pipeline on a video stream and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

This README is structured in a Q&A fashion, where each section is comprised of several questions or issues that we had to
tackle in order to meet the minimum requirements stated in [this rubric](https://review.udacity.com/#!/rubrics/513/view).

**NOTE**: You can download the vehicle dataset [here](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and the non-vehicles dataset [here](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip). These are the images that were used in this project.
## Writeup / README

#### I. _"Provide a Writeup / README that includes all the rubric points and how you addressed each one."_

This is the README. Keep reading to find out how we applied several cool computer vision techniques to solve the problem at hand ;)

## Histogram of Oriented Gradients (HOG)

#### I. _"Explain how (and identify where in your code) you extracted HOG features from the training images. Explain how you settled on your final choice of HOG parameters."_

In order to extract the HOG features that gave us the best results, we proceeded to explore the following parameter grid:

```
pix_per_cell: 8, 16
orientations: 8, 9, 24, 32
cell_per_block: 1, 2
color_spaces: HSV, YCrCb
```

The results of this exploration are in the `output_images/hog` directory.

We settled for the following parameters:
```
pix_per_cell: 8
orientations: 24
cell_per_block: 2
color_space: YcrCb
```

One of the important takeaways of this process is that the lightning variability was an issue to take into account. That's why we explored color spaces that usually deal well with variable lightning conditions, such as HSV and YCrCb. Also, we applied Gamma normalization over the images to diminish a bit more the effects of shadows and other illumination variations.

Here's the function we used to compute HOG features:

```
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
```

Here's an example of a car image:

![alt-tag](https://github.com/jesus-a-martinez-v/vehicle-detection/blob/master/output_images/hog/car_original.jpg)

Here's the same image converted to YCrCb color space:

![alt-tag](https://github.com/jesus-a-martinez-v/vehicle-detection/blob/master/output_images/hog/car_ycrcb.jpg)

Here's the HOG visualization:

![alt-tag](https://github.com/jesus-a-martinez-v/vehicle-detection/blob/master/output_images/hog/car_hog_colorspace=ycrcb_channel=0_pix_per_cell=8_orientation=24_cell_per_block=2.jpg)

Here's an example of a NON car image:

![alt-tag](https://github.com/jesus-a-martinez-v/vehicle-detection/blob/master/output_images/hog/non_car_original.jpg)

Here's the same image converted to YCrCb color space:

![alt-tag](https://github.com/jesus-a-martinez-v/vehicle-detection/blob/master/output_images/hog/non_car_ycrcb.jpg)

Here's the HOG visualization:

![alt-tag](https://github.com/jesus-a-martinez-v/vehicle-detection/blob/master/output_images/hog/non_car_hog_colorspace=ycrcb_channel=0_pix_per_cell=8_orientation=24_cell_per_block=2.jpg)


##### NOTE: Parameters

> We stored all the parameters used throughout the pipeline, the data, the classifier and the scaler in a pickled file called `db.p`, which as its name suggests, acted as our local
> database, reducing the number of parameters passed to each function. The code used to prepare the data as well as the parameters is in `data.py`. To download the exact `db.p` we used, click [here](https://drive.google.com/file/d/0B1SO9hJRt-hgWEN4bHMxaDhDZlU/view?usp=sharing) to get a zipped version of it.


#### II. _"Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them)."_

To train a classifier we used the code that's in `classifier.py`. At first, we trained several [`AdaBoostClassifiers`](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html) as well as [`RandomForestClassifiers`](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html), but the training time was too high while the accuracy gains where meaningless compared to a plain [`LinearSVC`](http://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html).

The features and the values used to extract them are:
```
{
   # Didn't use color features.
   'number_of_bins': 32,
   'histogram_features': False,
   
   # Didn't use spatial binning.
   'spatial_features': False, 
   'spatial_size': (32, 32),
   
   # We did use HOG features.
   'hog_features': True,
   'color_space': 'YCrCb',
   'orientations': 24,
   'pix_per_cell': 8,
   'cell_per_block': 2,
   'hog_channels': 'ALL',
   
   # Scales used in the sliding window technique
   'scales': (0.75, 1.5, 2, 2.25),
   
   # Number of frames used to smooth the detections, and to decrease the amount of false positives.
   'smoothing_factor': 5
}
```

This selection is the result of many trial and error iterations. The main trade-off was accuracy versus training/prediction time. So, that's why we kept the HOG features only, because the spatial and color features didn't improve much the accuracy of our model, but they greatly increased the image processing time.

To prepare the data we used the following function:
```
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
```

And to train the model, we used this:

```
def train_model(features, labels, test_proportion=0.20, seed=9991, save_model=True, location='./db.p'):
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
            'max_iter': (1000, 5000, 10000)
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
```

As we can see above, the first time we used a [`GridSearchClassifier`](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) to select the model parameters that provide the best performance. Once we find these parameters, we store them so we don't have to repeat the search all over again if we want to re-train the model.

Finally, we obtained an accuracy of 98.7% on the test set (which is the 20% of the training set).

## Sliding Window Search

#### I. _"Describe how (and identify where in your code) you implemented a sliding window search. How did you decide what scales to search and how much to overlap windows?"_

To improve the performance of the pipeline, instead of applying HOG over each window, we applied it once over the entire region of interest, and then subsampled that HOG featurized region at different scales. The scales selection was also a 
result of trial and error. Our rationale was to keep enough scales to detect farther and nearer cars, but not too much to bloat our pipeline with more windows that wouldn't improve its performance.
We settled with the following scales: `(0.75, 1.5, 2, 2.25)`

Also, we focused only on the right region of the frames (this is where the cars are in the test video). The function that performs the window search is:

```
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
        cells_per_step = 1  # Instead of overlap, define how many cells to step
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
```

And the one used to process each image (the actual pipeline) is:
```
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
```

This technique was extracted from [this amazing Q&A session held by Udacity](https://www.youtube.com/watch?v=P2zwrTM8ueA&feature=youtu.be).

#### II. _"Show some examples of test images to demonstrate how your pipeline is working. How did you optimize the performance of your classifier?"_

Here's an example of the windows found by the `find_cars()` function:
![alt-tag](https://github.com/jesus-a-martinez-v/vehicle-detection/blob/master/output_images/pipeline/test_6_raw_boxes.jpg)

Here's the associated heatmap:
![alt-tag](https://github.com/jesus-a-martinez-v/vehicle-detection/blob/master/output_images/pipeline/test_6_heatmap.jpg)

And here's the final image after merging the rectangles together:
![alt-tag](https://github.com/jesus-a-martinez-v/vehicle-detection/blob/master/output_images/pipeline/test_6_bboxes.jpg)

As we explained above, we decided to focus solely on the HOG features given that other spatial and color features didn't really improve that much the accuracy of the classifier, but impacted
heavily its performance. To deal with false positives, we first norrowed the search area to a region where the cars in the test video are most likely to appear (the right half below the horizon).
Then, to diminish the impact of false detections we kept heat maps of each processed frame and then thresholded the last 5 frames' heatmaps. This thresholding techique allowed us to get rid of detections in "cold" areas in the resulting average 
 heat map due to most likely being false positives.
 

## Video Implementation

#### I. _"Provide a link to your final video output. Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)"_

You can watch the result of processing a footage from a camera mounted on a car by clicking [here!](https://drive.google.com/file/d/0B1SO9hJRt-hgM25YR1N0NnFRb0k/view?usp=sharing) :).

#### II. _"Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes."_

To filter out false positives we kept a heat map of the detections in each frame. This heat map technique basically consist of taking a canvas (an image where all its pixels are set to zero) and adding "heat" (just increasing by one) those pixels within bounding boxes. So, more robust detections would have a lot of heat, while false positives should be "colder". To decrease the chance of keeping false positives, we averaged the heatmaps of the last 5 frames, and then applied a threshold of 5, which translates in pixels with heat below that threshold being zeroed out.

Here's the function used to add heat to a given heatmap:

```
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
```

And here's the function that applies a threshold toa heatmap:

```
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
```

Finally, to merge several detections in a particular area of the image, we used the handy [`label()`](https://docs.scipy.org/doc/scipy-0.16.0/reference/generated/scipy.ndimage.measurements.label.html) function in the SciPy library. Here's where we used it:

```
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
```

## Discussion

#### I. _"Briefly discuss any problems / issues you faced in your implementation of this project. Where will your pipeline likely fail? What could you do to make it more robust?"_

One of the disadvantages of the current pipeline is that it isn't very customizable. 
For instance, the look up area is skewed to the right because there's where the cars are in the test video, so a car
in another location would not be properly identified. Also, the pipeline has a hard time identifying bright colored objects (such as the white car). It may be necessary to add more features related to color and pixels distribution. 
 
Even when our classifier reached a 98.7% accuracy on the test set, it outputs more false positives than desired.
More training with a bigger or extended dataset is a must to improve the overall performance of the pipeline. Also, as stated in the previous paragraph, further exploration of color features (other color spaces, for instance) should improve the results.

Although computer vision provides a really powerful set of tools, I find the fine-tuning process very exhausting and 
I am not so sure if this could scale well to a production environment, whereas a neural network, at least to me, has 
fewer knobs to tweak and converges to a more robust solution faster. The downside of this latter approach, of course, 
is that we have no control over how the network learns, which wraps all the process with a "magic" aura, 
difficulting the debugging activities. On the other hand, computer vision passes all the responsibility to us, and while 
this situation greatly increases the complexity, knowing exactly why something works (or not) is extremely useful. 
Perhaps the best solution lies somewhere in between these two worlds! :)
