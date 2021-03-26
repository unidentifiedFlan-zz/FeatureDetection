import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from cycler import cycler
import EdgeDetection
import PointDetection
import FeatureDescription
import FeatureMatching
import GlobalTransformations

#Shi Tomasi 1994 Good features to track
#
#Use Newton-Raphson to estimate affine transformation
#parameters
#Monitor ongoing quality of a feature since creation
#by using dissimilarity metric -- rms residue between
#initial and current measure of feature
#
#

#Real-time feature tracking -LePetit 2004
#
#Initialise with first frame:
#for each feature found, using a fairly high
#threshold to reduce misclassification, create a set of views
#using deformation of the surrounding patch.
#For each set create a descriptor for each deformed image
#using PCA.
#For each set perform k-means clustering on the descriptors
#to represent each set using k vectors
#
#For face tracking, texture map the first frame's face onto
#a 3D mesh and create a viewset by deforming the mesh
#
#For each subsequent frame, find features
#create PCA descriptors for surrounding patches.
#Classify each feature to a viewset by comparing
#the descriptor with the k mean descriptors.
#Assign the feature to the nearest neighbour set.

def transform_to_black_white(image):
    return image.convert('L')

def find_edge_features(image, padding):
    edgeDetector = EdgeDetection.EdgeDetector()
    edgeFeatures = edgeDetector.find_features(image, padding)
    features = []
    for i in edgeFeatures:
        features.append([i.x, i.y])
    FD = FeatureDescription.FeatureDescription()
    descriptors = FD.SIFT(features, image)
    return features, descriptors

def find_point_features(image):
    detector = PointDetection.GaussianPointDetector(image)
    features = detector.find_features(detector.scalar_measure.Anandan)
    FD = FeatureDescription.FeatureDescription()
    descriptors = FD.SIFT(features, image)
    return features, descriptors

def plot_matches(original_image, original_features, image, matches):
    plt.subplot(2, 1, 1)
    plt.imshow(Image.fromarray(original_image), cmap='gray')
    for i, f in enumerate(original_features):
        if matches.__contains__(i):
            plt.scatter(f[1], f[0], marker="+")
            plt.annotate(i, xy=(f[1], f[0]))
        if i > 500:
            break
    plt.subplot(2, 1, 2)
    plt.imshow(Image.fromarray(image), cmap='gray')
    for i, indices in matches.items():
        plt.scatter([indices[1]], [indices[0]], marker="+")
        plt.annotate(i, xy=(indices[1], indices[0]))
        if i > 500:
            break

def main_feature_tracking():
    path = "ComputerVision/videoResources/headtracker_sequences/seq_villains1/"
    plt.rc('axes', prop_cycle=(cycler('color', ['r', 'g', 'b'])))

    padding = 50
    original_descriptors = []
    for i in range(3):
        plt.subplot(1, 3, int(i%3) +1)
        file = "img00" + str(i) + ".bmp"
        image = Image.open(path + file)
        image_arr = prepare_image(image, padding, [0,0])
        plt.imshow(Image.fromarray(image_arr), cmap='gray')
        features, descriptors = find_point_features(image_arr)
#        features, descriptors = find_edge_features(image_arr, padding)
        if i == 0:
            original_features = features
            original_descriptors = descriptors
            for i, f in enumerate(original_features):
                plt.scatter(f[1], f[0], marker="+")
                plt.annotate(i, xy=(f[1], f[0]))
                if i > 500:
                    break
        else:
            matches = FeatureMatching.nearest_neighbour(original_descriptors, features, descriptors)
            for i, indices in matches.items():
                plt.scatter([indices[1]], [indices[0]], marker="+")
                plt.annotate(i, xy=(indices[1], indices[0]))
                if i > 500:
                    break

        if i%3 == 0:
            plt.show()

def prepare_image(image, padding, translation):
    BWImage = transform_to_black_white(image)
    BWImage = np.asarray(BWImage)
    padded_array = GlobalTransformations.addPadding(BWImage, padding)
    return GlobalTransformations.translate(padded_array, translation)

def main_feature_matching():
    path = "ComputerVision/FeatureDetection/"
    files = ["mountain1.png", "mountain2.png"]
    translations = [[0,0], [0, 0]]
    padding = 50
    plt.rc('axes', prop_cycle=(cycler('color', ['r', 'g', 'b'])))

    original_features = []
    original_descriptors = []
    for i, f in enumerate(files):
        image = Image.open(path + f)
        image_arr = prepare_image(image, padding, translations[i])
        if i == 0:
            original_image_arr = image_arr
        features, descriptors = find_point_features(image_arr)
#        features, descriptors = find_edge_features(image_arr, padding)
        if i == 0:
            original_features = features
            original_descriptors = descriptors
        else:
            matches = FeatureMatching.nearest_neighbour_distance_ratio(original_descriptors, features, descriptors)
            plot_matches(original_image_arr, original_features, image_arr, matches)

    plt.show()

if __name__ == '__main__':
    main_feature_matching()