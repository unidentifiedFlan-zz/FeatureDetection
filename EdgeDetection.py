from Convolution import Convolution
import ImagePyramid
import numpy as np
import math

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

class Gradient:
    def __init__(self):
        self.normal = Point(0,0) #rename to vector
        self.magnitude = 0

class Edgel:
    def __init__(self, crossing1, crossing2, values1, values2):
        self.end = []
        self.end.append(self.calculateEnd(crossing1, values1))
        self.end.append(self.calculateEnd(crossing2, values2))
        gradient1 = self.calculateGradient(crossing1, values1)
        gradient2 = self.calculateGradient(crossing2, values2)
        self.calculateEdgelGradient(gradient1, gradient2)
        mid = self.calculateMidpoint()
        self.x = mid.x
        self.y = mid.y

    def calculateEnd(self, positions, values):
        sign0 = 1 if values[0] > 0 else -1
        sign1 = 1 if values[1] > 0 else -1
        s_diff = sign0 - sign1
        return Point((positions[1].x*sign0 - positions[0].x*sign1)/s_diff, (positions[1].y*sign0 - positions[0].y*sign1)/s_diff)

    def calculateMidpoint(self):
        x_diff = self.end[1].x - self.end[0].x
        y_diff = self.end[1].y - self.end[0].y
        mid = Point(0,0)
        mid.x = self.end[0].x + x_diff if x_diff > 0 else self.end[1].x - x_diff
        mid.y = self.end[0].y + y_diff if y_diff > 0 else self.end[1].y - y_diff
        return mid

    def calculateEdgelGradient(self, gradient1, gradient2):
        gradient = Gradient()
        if gradient1.normal.x == gradient2.normal.x:
            gradient.normal.x = gradient1.normal.x
            if gradient.normal.x == 0:
                if gradient1.normal.y * gradient2.normal.y < 0:
                    diff = gradient2.magnitude - gradient1.magnitude
                    gradient.magnitude = -diff if gradient1.magnitude > gradient2.magnitude else diff
                    gradient.normal.y = gradient1.normal.y if gradient1.magnitude > gradient2.magnitude else gradient2.normal.y
                else:
                    gradient.normal.y = gradient1.normal.y
                    gradient.magnitude = gradient1.magnitude if gradient1.magnitude > gradient2.magnitude else gradient2.magnitude
            else:
                gradient.normal.y = 0;
                gradient.magnitude = gradient1.magnitude if gradient1.magnitude > gradient2.magnitude else gradient2.magnitude
        elif gradient1.normal.x == -gradient2.normal.x:
            gradient.normal.y = 0;
            diff = gradient2.magnitude - gradient1.magnitude
            gradient.magnitude = -diff if diff < 0 else diff
            gradient.normal.x = gradient1.normal.x if diff < 0 else gradient2.normal.x
        else:
            normFactor = 1/(math.sqrt(2))
            gradient.normal.x = normFactor*gradient2.normal.x if gradient1.normal.x == 0 else normFactor*gradient1.normal.x
            gradient.normal.y = normFactor*gradient2.normal.y if gradient1.normal.y == 0 else normFactor*gradient1.normal.y
            gradient.magnitude = gradient1.magnitude if gradient1.magnitude > gradient2.magnitude else gradient2.magnitude

        return gradient

    def calculateGradient(self, crossing, values):
        gradient = Gradient()
        gradient.magnitude = abs(values[1] - values[0])
        gradient.normal = self.calculateNormal(crossing, values[0], values[1])
        return gradient

    def calculateNormal(self, crossing, value1, value2):
        normal = Point(0,0)
        imin = 0
        imax = 1
        if value2 < value1:
            imin=1
            imax=0
        if crossing[0].x == crossing[1].x:
            normal.x = 0
            normal.y = crossing[imax].y - crossing[imin].y
        else:
            normal.x = crossing[imax].x - crossing[imin].x
            normal.y = 0
        return normal

class EdgeDetector:
    def find_features(self, image, padding_width):
        gaussian_kernel = 0.0625*np.array([1, 4, 6, 4, 1])
        self.padding_width = padding_width
        cpy = np.asarray(image).copy()
        cpy = self.even_shape(cpy)
        blurred_image = Convolution(gaussian_kernel).convolve_separable(cpy)
        pyramid = ImagePyramid.Gaussian_Pyramid(blurred_image)
        interpolatedP = ImagePyramid.upscale_image(pyramid[1])
        interpolatedP = Convolution(gaussian_kernel).convolve_separable(interpolatedP)
        s = blurred_image - interpolatedP

        return self.find_edgels(s)

    def even_shape(self, array):
        width = array.shape[0]
        height = array.shape[1]
        if width % 2 == 1:
            width = width -1
        if height % 2 == 1:
            height = height -1
        return array[0:width,0:height]

    def find_edgels(self, image):
        edgels = []
        for x in range(image.shape[0]):
            for y in range(image.shape[1]):
                crossings = self.find_crossings(image, Point(x, y))
                if len(crossings) == 2:
                    values1 = []
                    values2 = []
                    values1.append(image[crossings[0][0].x, crossings[0][0].y])
                    values1.append(image[crossings[0][1].x, crossings[0][1].y])
                    values2.append(image[crossings[1][0].x, crossings[1][0].y])
                    values2.append(image[crossings[1][1].x, crossings[1][1].y])
                    edgels.append(Edgel(crossings[0], crossings[1], values1, values2))
        return edgels


    def find_crossings(self, image, pixel):
        crossings = []
        threshold = -10
        if pixel.x > image.shape[0] -2 or pixel.y > image.shape[1] -2:
            return crossings
        if pixel.x < self.padding_width or pixel.x > image.shape[0] - self.padding_width:
            return crossings
        if pixel.y < self.padding_width or pixel.y > image.shape[1] - self.padding_width:
            return crossings

        if image[pixel.x, pixel.y]*image[pixel.x+1, pixel.y] < threshold:
            crossings.append([Point(pixel.x, pixel.y), Point(pixel.x+1, pixel.y)])
        if image[pixel.x, pixel.y]*image[pixel.x, pixel.y+1] < threshold:
            crossings.append([Point(pixel.x, pixel.y), Point(pixel.x, pixel.y+1)])
        if image[pixel.x+1, pixel.y]*image[pixel.x+1, pixel.y+1] < threshold:
            crossings.append([Point(pixel.x+1, pixel.y), Point(pixel.x+1, pixel.y+1)])
        if image[pixel.x, pixel.y+1]*image[pixel.x+1, pixel.y+1] < threshold:
            crossings.append([Point(pixel.x, pixel.y+1), Point(pixel.x+1, pixel.y+1)])
        return crossings
