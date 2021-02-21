import numpy as np
import math
import scipy.stats as st

class Convolution:
    def __init__(self, kernel_h):
        self.kernel = kernel_h
        self.image = np.zeros((0,0))

    def gaussian_kernel(kernlen=16, nsig=3):
        x = np.linspace(-nsig, nsig, kernlen + 1)
        kernel = np.diff(st.norm.cdf(x))
        return kernel / kernel.sum()

    def convolve_horizontally(self):
        width = self.image.shape[0]
        conv_h = self.image.copy()
        if self.image.ndim < 3:
            for i in range(width):
                cpy_arr = conv_h[i, :]
                conv_h[i, :] = np.convolve(cpy_arr, self.kernel, 'same')
        else:
            channels = self.image.shape[2]
            for c in range(channels):
                for i in range(width):
                    cpy_arr = conv_h[i, :, c]
                    conv_h[i, :, c] = np.convolve(cpy_arr, self.kernel, 'same')
        return conv_h

    def convolve_vertically(self):
        height = self.image.shape[1]
        conv_v = self.image.copy()
        if self.image.ndim < 3:
            for j in range(height):
                cpy_arr = conv_v[:, j]
                conv_v[:, j] = np.convolve(cpy_arr, self.kernel, 'same')
        else:
            channels = self.image.shape[2]
            for c in range(channels):
                for j in range(height):
                    cpy_arr = conv_v[:, j, c]
                    conv_v[:, j, c] = np.convolve(cpy_arr, self.kernel, 'same')

        return conv_v

    def convolve_separable(self, image):
        self.image = image.copy()
        self.image = self.convolve_horizontally()
        self.image = self.convolve_vertically()
        return self.image

    def convolve_directonally(self, image, theta):
        self.image = image.copy()
        conv_h = self.convolve_horizontally()
        conv_v = self.convolve_vertically()
        return math.cos(theta)*conv_h + math.sin(theta)*conv_v
