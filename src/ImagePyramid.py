import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sys import exit
from Convolution import Convolution

class LaplacianPyramid:
    def __init__(self, image):
        self.kernel = 0.0625*np.array([1, 4, 6, 4, 1])
        self.image = image.copy()
        self.pyramid = self.construct()

    def construct(self):
        laplacian_list = []
        gaussian_list = Gaussian_Pyramid(self.image)
        list_length = len(gaussian_list)

        for j in range(list_length-1, 0, -1):
            upscaled_arr = upscale_image(gaussian_list[j])
            upscaled_arr = Convolution(self.kernel).convolve_separable(upscaled_arr)
            laplacian_arr = np.subtract(gaussian_list[j-1], upscaled_arr)
            laplacian_list.append(laplacian_arr)

        return laplacian_list

def upscale_image(image):
    factor = 2
    if image.ndim > 2:
        i_width, i_height, i_channels = image.shape
    else:
        i_width, i_height = image.shape
    o_width = factor * i_width
    o_height = factor * i_height
    if image.ndim > 2:
        new_img = np.zeros((o_width, o_height, i_channels))
    else:
        new_img = np.zeros((o_width, o_height))

    for x in range(o_width):
        for y in range(o_height):
            new_img[x][y] = image[int(x/factor)][int(y/factor)]
    return new_img

def downscale_image(image):
    factor = int(2)
    if image.ndim > 2:
        i_width, i_height, i_channels = image.shape
    else:
        i_width, i_height = image.shape
    o_width = int(i_width / factor)
    o_height = int(i_height / factor)
    if image.ndim > 2:
        new_img = np.zeros((o_width, o_height, i_channels))
    else:
        new_img = np.zeros((o_width, o_height))

    for x in range(o_width):
        for y in range(o_height):
            new_img[x][y] = image[factor * x][factor * y]
    return new_img

def Gaussian_Pyramid(image):
    kernel = 1 / 16 * np.array([1, 4, 6, 4, 1])
    width = image.shape[0]
    height = image.shape[1]
    if width < height:
        min_dim = width
    else:
        min_dim = height

    gaussian_list = [image]
    decimated_arr = image
    i = min_dim

    kernel_width = kernel.size
    while i > kernel_width:
        blurred_arr = Convolution(kernel).convolve_separable(decimated_arr)
        decimated_arr = downscale_image(blurred_arr)
        gaussian_list.append(decimated_arr)
        i = i / 2

    return gaussian_list

def reconstruct_from_laplacian_pyramid(pyramid):
    kernel = 1 / 16 * np.array([1, 4, 6, 4, 1])
    n = len(pyramid)
    current_img = pyramid[0]
    for i in range(1, n):
        upscaled_img = upscale_image(current_img)
        upscaled_img = Convolution(kernel).convolve_separable(upscaled_img)
        current_img = upscaled_img + pyramid[i]

    return current_img

def main():
    bilinear_kernel = 0.25 * np.array([1, 2, 1])
    binomial_kernel = 1 / 16 * np.array([1, 4, 6, 4, 1])
    bicubic_kernel = 1 / 16 * np.array([-1, 0, 5, 8, 5, 0, -1])
    windowed_sinc_kernel = np.array([0.0, -0.0153, 0.0, 0.2684, 0.4939, 0.2684, 0.0, -0.0153, 0.0])

    dir = "ComputerVision/image-blending/"
    apple = dir + "apple.jpg"
    orange = dir + "orange.jpg"
#    mask = dir + "vertical-mask.jpg"

    apple_im = Image.open(apple).convert('RGB')
    apple_arr = np.asarray(apple_im)
    laplacian_apple_list = np.asarray(LaplacianPyramid(apple_arr).pyramid)

    orange_im = Image.open(orange).convert('RGB')
    orange_arr = np.asarray(orange_im)
    laplacian_orange_list = np.asarray(LaplacianPyramid(orange_arr).pyramid)

    combined_laplacian_pyramid = []
    for la, lb in zip(laplacian_apple_list, laplacian_orange_list):
        rows = la.shape[0]
        cols = la.shape[1]
        ls = np.hstack((la[:, 0:int(cols / 2)], lb[:, int(cols / 2):]))
        combined_laplacian_pyramid.append(ls)

    final_arr = reconstruct_from_laplacian_pyramid(combined_laplacian_pyramid)
    final_arr = np.clip(final_arr, a_min=0, a_max=255)
    final_im = Image.fromarray(final_arr.astype('uint8')).convert('RGB')
    final_im.save(dir + "output.jpg")

    plt.imshow(final_im)
    plt.show()


if __name__ == '__main__':
    main()