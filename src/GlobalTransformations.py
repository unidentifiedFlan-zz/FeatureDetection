import numpy as np

def translate(image, translation):
    arr = np.zeros( image.shape )
    if translation[0] == 0 and translation[1] == 0:
        return image
    if translation[0] == 0:
        arr[translation[0]:, translation[1]:] = image[:, :-translation[1]]
    elif translation[1] == 0:
        arr[translation[0]:, translation[1]:] = image[:-translation[0], :]
    else:
        arr[translation[0]:, translation[1]:] = image[:-translation[0], :-translation[1]]

    #Less likely to get such sharp contrasts with black,
    #for a typical image, than with white
    blackRows = np.zeros((translation[0], image.shape[1]))
    blackCols = np.zeros((image.shape[0], translation[1]))
    arr[:translation[0], :] = blackRows
    arr[:, :translation[1]] = blackCols

    return arr

def addPadding(image, width):
    arr = np.zeros((image.shape[0] + 2*width, image.shape[1] + 2*width))
    arr[width:image.shape[0] + width, width: image.shape[1]+width] = image
    return arr
