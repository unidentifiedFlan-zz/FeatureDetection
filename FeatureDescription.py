import numpy as np
from sklearn.decomposition import PCA
import math
from Convolution import Convolution

class Feature:
    def __init__(self, feature, descriptor):
        self.x = feature[0]
        self.y = feature[1]
        self.pcaSiftDescriptor = descriptor

class FeatureDescription:
    def most_significant_components(self, feature_vec):
        pca = PCA(n_components=2)
        pca.fit(feature_vec)
        return pca.singular_values_

    def PCA_SIFT(self, features, image):
        kernel = np.array([3, 5, 0, -5, -3])
        feature_vectors = []

        cpy = np.asarray(image).copy()
        grad_x = Convolution(kernel).convolve_directonally(cpy, 0)
        cpy = np.asarray(image).copy()
        grad_y = Convolution(kernel).convolve_directonally(cpy, math.pi/2)

        neighbourhood_size = 19
        size_x = cpy.shape[0]
        size_y = cpy.shape[1]

        for f in features:
            xmin = f[0]-neighbourhood_size
            xmin = xmin if xmin >= 0 else 0
            xmax = f[0]+neighbourhood_size
            xmax = xmax if xmax < size_x else size_x - 1
            ymin = f[1]-neighbourhood_size
            ymin = ymin if ymin >= 0 else 0
            ymax = f[1]+neighbourhood_size
            ymax = ymax if ymax < size_y else size_y - 1

            feature_vec = np.asarray(grad_x[xmin:xmax,ymin:ymax]) + np.asarray(grad_y[xmin:xmax, ymin:ymax])
            feature_vectors.append(self.most_significant_components(feature_vec))

        return feature_vectors

    def SIFT(self, features, image):
        neighbourhood_size = 16
        kernel = np.array([3, 5, 0, -5, -3])
        cpy = np.asarray(image).copy()
        grad_x = Convolution(kernel).convolve_directonally(cpy, 0)
        grad_x = Convolution(Convolution.gaussian_kernel(neighbourhood_size)).convolve_separable(grad_x)
        cpy = np.asarray(image).copy()
        grad_y = Convolution(kernel).convolve_directonally(cpy, math.pi/2)
        grad_y = Convolution(Convolution.gaussian_kernel(neighbourhood_size)).convolve_separable(grad_y)

        feature_vectors = []
        for feature in features:
            min_x, max_x, min_y, max_y = self.neighbourhood_boundaries(cpy, feature, neighbourhood_size)
            histogram =[]
            for i in range(4):
                for j in range(4):
                    x_low = int(min_x + (max_x - min_x) * int(i / 4))
                    x_high = int(min_x + (max_x - min_x) * int((i+1) / 4))
                    y_low = int(min_y + (max_y - min_y) * int(j / 4))
                    y_high = int(min_y + (max_y - min_y) * int((j+1) / 4))
                    histogram.append(self.SIFT_histogram(grad_x[x_low: x_high, y_low: y_high], grad_y[x_low: x_high, y_low: y_high]))

            feature_vectors.append(self.normalise_feature_vector(histogram))

        return feature_vectors

    def SIFT_histogram(self, gradients_x, gradients_y):
        bins = np.zeros((8))
        for i in range(gradients_x.shape[0]):
            for j in range(gradients_x.shape[1]):
                grad_x = gradients_x[i,j]
                grad_y = gradients_y[i,j]
                theta = self.theta(grad_x, grad_y)
                bin = self.determine_bin(theta)
                bins[bin] += (1/math.sqrt(2))*(grad_x*grad_x + grad_y*grad_y)
        return bins

    def theta(self, grad_x, grad_y):
        if grad_x == 0 and grad_y == 0:
            return 0
        if grad_x > 0:
            return 90*grad_y/grad_x
        elif grad_x == 0:
            return 90*grad_y/abs(grad_y)
        elif grad_y > 0:
            return 180*(grad_y/abs(grad_y)) - (90*grad_y/grad_x)
        else:
            return 180

    def determine_bin(self, theta):
        if theta >=0:
            if theta < 45:
                return 0
            elif theta < 90:
                return 1
            elif theta < 135:
                return 2
            else:
                return 3
        else:
            if theta > -45:
                return 4
            elif theta > -90:
                return 5
            elif theta > -135:
                return 6
            else:
                return 7

    def normalise_feature_vector(self, histogram):
        sum = np.linalg.norm(histogram)
        histogram /= sum
        histogram = np.clip(histogram, 0, 0.2)
        sum = np.linalg.norm(histogram)
        histogram /= sum

        return histogram

    def neighbourhood_boundaries(self, image, feature, neighbourhood_size = 16):
        size_x = image.shape[0]
        size_y = image.shape[1]
        extent = int(neighbourhood_size/2)
        if feature[0] - extent >= 0:
            if feature[0] + extent < size_x:
                min_x = feature[0] - int(extent)
                max_x = feature[0] + extent - 1
            else:
                min_x = feature[0] - extent
                max_x = size_x - 1
        else:
            min_x = 0
            max_x = feature[0] + extent - 1

        if feature[1] - extent >= 0:
            if feature[1] + extent < size_x:
                min_y = feature[1] - extent
                max_y = feature[1] + extent - 1
            else:
                min_y = feature[1] - extent
                max_y = size_y - 1
        else:
            min_y = 0
            max_y = feature[1] + extent - 1

        return min_x, max_x, min_y, max_y