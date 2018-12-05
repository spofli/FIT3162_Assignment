import os
import time
import cv2
import numpy as np
from svm_skikit import svm
from knn_skikit import knn
from CCA import connected_components
from sklearn.model_selection import KFold
from get_lbp_feature import get_lbp_feature
from neighbour_points import neighbour_points

if __name__ == "__main__":
    start_clock = time.strftime("%m/%d/%Y %H:%M:%S")
    start_timer = time.time()
    print(start_clock)

    # Get directory containing images
    # comp_fig_path = "./CompoundFigureDetectionTrainingDataset/COMP/"
    # non_comp_fig_path = "./CompoundFigureDetectionTrainingDataset/NOCOMP/"
    comp_fig_path = "./fine_tuning/COMP/"
    non_comp_fig_path = "./fine_tuning/NOCOMP/"
    # Get filenames of images
    comp_fig_filenames = os.listdir(comp_fig_path)
    non_comp_fig_filenames = os.listdir(non_comp_fig_path)

    # Get data from images
    im_dataset = []
    im_classification = []
    for i in range(len(comp_fig_filenames)):
        im_dataset.append(cv2.imread(comp_fig_path + comp_fig_filenames[i], 0))
        im_classification.append(1)
    for i in range(len(non_comp_fig_filenames)):
        im_dataset.append(cv2.imread(non_comp_fig_path + non_comp_fig_filenames[i], 0))
        im_classification.append(-1)
    # np.save("image_data", im_dataset)
    # np.save("class", im_classification)

    # im_dataset = np.load("image_data.npy")
    # im_classification = np.load("class.npy")

    # Deciding LBP operator configuration
    maptype = "riu2"
    radius = 1
    resize_width = 300
    resize_height = 300
    arrangement_x = [3, 2]
    arrangement_y = [2, 3]
    config = 1
    # Get neighbour points for given radius
    neighbour_info = neighbour_points(radius)
    print("Processing image data to get feature descriptors")
    t0 = time.time()
    # Extract LBP/ CLBP features from each image in dataset
    lbp_features = [None] * len(im_dataset)
    clbp_features = [None] * len(im_dataset)
    #cca_features = [None] * len(im_dataset)
    for i in range(len(im_dataset)):
        lbp_features[i], clbp_features[i] = get_lbp_feature(im_dataset[i], neighbour_info, resize_width, resize_height,
                                                            arrangement_x, arrangement_y, maptype=maptype,
                                                            radius=radius)
        #cca_features[i] = connected_components(im_dataset[i])
        if (i - 1) % 1 == 0:
            print("%d/%d" % (i, len(im_dataset)))
    print()
    print("Time taken: %.2fs" % (time.time() - t0))
    #np.save("features_lbp_%s_r%d_v%d" % (maptype, radius, config), lbp_features)
    #np.save("features_clbp_%s_r%d_v%d" % (maptype, radius, config), clbp_features)
    #np.save("features_cca", cca_features)
    # lbp_features = np.load("features_lbp_%s_r%d_v%d.npy" % (maptype, radius, config))
    # clbp_features = np.load("features_clbp_%s_r%d_v%d.npy" % (maptype, radius, config))
    # cca_features = np.load("features_cca.npy")

    for ftype in range(2):
        if ftype == 0:
            features = lbp_features
        else:
            features = clbp_features

        print("Starting k-fold cross validation")
        # Initialize cross-validation k-folds training/test sets
        k_fold_iteration = 1
        svm_accuracy_vals = []
        knn_accuracy_vals = []
        svm_time = 0
        knn_time = 0
        k_folds = KFold(n_splits=10, shuffle=True, random_state=1)
        for train_indexes, test_indexes in k_folds.split(features):
            #print("Fold:", k_fold_iteration)
            # Getting feature data from training and testing images
            train_data = []
            train_class = []
            for i in train_indexes:
                feature_diff = features[i]
                train_data.append(feature_diff)
                train_class.append(im_classification[i])
            test_data = []
            test_class = []
            for i in test_indexes:
                feature_diff = features[i]
                test_data.append(feature_diff)
                test_class.append(im_classification[i])
            t1 = time.time()
            # Train and test with SVM
            accuracy = svm(train_data, train_class, test_data, test_class)
            svm_accuracy_vals.append(accuracy)
            t2 = time.time()
            svm_time += t2 - t1
            #print("SVM time: %.2fs, Accuracy: %.2f" % ((t2 - t1), accuracy))
            t1 = time.time()
            # Train and test with KNN
            accuracy = knn(train_data, train_class, test_data, test_class)
            knn_accuracy_vals.append(accuracy)
            t2 = time.time()
            knn_time += t2 - t1
            #print("KNN time: %.2fs, Accuracy: %.2f" % ((t2 - t1), accuracy))
            k_fold_iteration += 1
        print()
        mean_svm = float(np.mean(svm_accuracy_vals))
        mean_knn = float(np.mean(knn_accuracy_vals))
        print("SVM Accuracy: %.2f, Time: %.2f\nKNN Accuracy: %.2f, Time: %.2f\n" % (
            mean_svm, svm_time, mean_knn, knn_time))
    print("start:", start_clock, "| end:", time.strftime("%m/%d/%Y %H:%M:%S"))
    print("Time taken: %.2fs" % (time.time() - start_timer))

    # image = cv2.imread('aaaa.jpg', cv2.IMREAD_GRAYSCALE)
    # print(image)
    # lbp_image = lbp(image, 8, 1)
    # cv2.imshow("lbp", lbp_image)
    # cv2.imshow("original", image)
    # cv2.waitKey(0)
