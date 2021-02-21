import numpy as np
import math
from enum import Enum
from Convolution import Convolution

class GaussianPointDetector:
    class scalar_measure(Enum):
        Anandan = 1
        Forstner_Harris = 2

    def __init__(self, image):
        self.features = []
        self.gaussian_deriv_kernel = np.array([3, 5, 0, -5, -3])
        self.gaussian_kernel = 0.0625*np.array([1, 4, 6, 4, 1])

        self.image = np.asarray(image)
        self.auto_correlation_surface = np.zeros(self.image.shape)
        self.measure_surface = np.zeros(self.image.shape)
        self.gauss_x = np.zeros(self.image.shape)
        self.gauss_y = np.zeros(self.image.shape)

    def find_features(self, scalar_measure):
        # 1) compute the horizontal and vertical image derivatives via Convolution
        # 2) compute outer products of derivative images
        # 3) convolve each of the outer products with a larger gaussian
        # 4) compute scalar interest measure
        # 5) Find local maxima above given threshold and report as a found feature

        self.compute_directional_derivatives()
        self.generate_autocorrelation_matrix() # This is an image of autocorrelation matrices
        self.compute_scalar_measure(scalar_measure)
        self.mark_features()
        return self.features

    def compute_directional_derivatives(self):
        self.gauss_x = Convolution(self.gaussian_deriv_kernel).convolve_directonally(self.image, 0)
        self.gauss_y = Convolution(self.gaussian_deriv_kernel).convolve_directonally(self.image, math.pi/2)


    def generate_autocorrelation_matrix(self):
        self.compute_gradient_products(self.gauss_x, self.gauss_y)

        #convolve with gaussian weight
        wgtd_grad_x_sq = Convolution(Convolution.gaussian_kernel(10, 2)).convolve_separable(self.grad_x_sq)
        wgtd_grad_y_sq = Convolution(Convolution.gaussian_kernel(10, 2)).convolve_separable(self.grad_y_sq)
        wgtd_cross = Convolution(Convolution.gaussian_kernel(10, 2)).convolve_separable(self.cross)
        self.stitch_weighted_grad_products(wgtd_grad_x_sq, wgtd_grad_y_sq, wgtd_cross)


    def compute_gradient_products(self, grad_x, grad_y):
        self.grad_x_sq = grad_x * grad_x
        self.grad_y_sq = grad_y * grad_y
        self.cross = grad_x * grad_y

    def stitch_weighted_grad_products(self, wgtd_grad_x_sq, wgtd_grad_y_sq, wgtd_cross):
        self.auto_correlation_surface = np.zeros(wgtd_grad_x_sq.shape, dtype=object)

        for x in range(wgtd_grad_x_sq.shape[0]):
            for y in range(wgtd_grad_x_sq.shape[1]):
                auto_correlation_matrix = np.zeros((2, 2))
                auto_correlation_matrix[0, 0] = wgtd_grad_x_sq[x, y]
                auto_correlation_matrix[0, 1] = wgtd_cross[x, y]
                auto_correlation_matrix[1, 0] = wgtd_cross[x, y]
                auto_correlation_matrix[1, 1] = wgtd_grad_y_sq[x, y]

                self.auto_correlation_surface[x, y] = auto_correlation_matrix

    def compute_scalar_measure(self, scalar_measure):
        self.measure_surface = np.zeros(self.auto_correlation_surface.shape)
        for x in range(self.auto_correlation_surface.shape[0]):
            for y in range(self.auto_correlation_surface.shape[1]):
                eigen_values = np.linalg.eigvals(self.auto_correlation_surface[x,y])
                small_eigen = np.min(eigen_values)
                large_eigen = np.max(eigen_values)
                if scalar_measure == scalar_measure.Anandan:
                    self.measure_surface[x, y] = small_eigen
                elif scalar_measure == scalar_measure.Forstner_Harris:
                    alpha = 0.05
                    self.measure_surface[x, y] = small_eigen - alpha*large_eigen

    def mark_features(self):
        #find local maxima above threshold
        #mark on original image
        neighbourhood_size = 3
        threshold = 0
        size_x = self.measure_surface.shape[0]
        size_y = self.measure_surface.shape[1]

        maxima = np.argwhere(self.measure_surface > threshold)

        local_maxima = []
        for indices in maxima:
            min_x = indices[0] - neighbourhood_size if indices[0]-neighbourhood_size >= 0 else 0
            max_x = indices[0] + neighbourhood_size if indices[0]+neighbourhood_size < size_x else size_x -1
            min_y = indices[1] - neighbourhood_size if indices[1]-neighbourhood_size >= 0 else 0
            max_y = indices[1] + neighbourhood_size if indices[1]+neighbourhood_size < size_y else size_y - 1

            neighbourhood = self.measure_surface[min_x : max_x, min_y: max_y]

            local_max_col = np.max(neighbourhood, axis=0)
            local_max_h = np.argmax(local_max_col)
            local_max_v = np.argmax(neighbourhood[:, local_max_h])

            if indices[0] == local_max_h + min_x and indices[1] == local_max_v + min_y:
                local_maxima.append(indices)

        self.features = local_maxima
