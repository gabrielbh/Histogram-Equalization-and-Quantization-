import numpy as np
from skimage.color import rgb2gray
from scipy.misc import imread, imsave, imresize
import matplotlib.pyplot as plt


rgb2yiqMatrix = [[0.299, 0.587, 0.114], [0.596, -0.275, -0.321], [0.212, -0.523, 0.311]]
Matrix = np.array(rgb2yiqMatrix)
GREY_LEVEL_MAX_VAL = 256
INPUT_ERROR_MSG = "INPUT ERROR"

"""
function which reads an image file and converts it into a given representation.
This function returns an image, normalized to the range [0, 1].
"""
def read_image(filename, representation):
    im = imread(filename).astype(np.float64) / (GREY_LEVEL_MAX_VAL - 1)
    if (representation == 1):
        im_g = rgb2gray(im)
        return im_g
    return im


"""
function to utilize read_image to display an image in a given representation.
"""
def imdisplay(filename, representation):
    im = read_image(filename, representation)
    plt.imshow(im, cmap='gray' if representation == 1 else None)
    plt.show()


"""
function that transform an RGB image into the YIQ color space.
"""
def rgb2yiq(imRGB):
    yiq = imRGB.dot(Matrix.T)
    return yiq


"""
function that transform an YIQ image into the RGB color space.
"""
def yiq2rgb(imYIQ):
    invMat = np.linalg.inv(Matrix)
    rgb = imYIQ.dot(invMat.T)
    return rgb



'''
returns true if the image in path is RGB.
'''
def rgb_scale(path):
    return len(path.shape) == 3


"""
function that performs histogram equalization of a given grayscale or RGB image.
"""
def histogram_equalize(im_orig):
    chanel = im_orig
    if rgb_scale(im_orig):
        yiq = rgb2yiq(np.array(im_orig))
        chanel = yiq[:, :, 0]
    hist_orig, bins = np.histogram((chanel * (GREY_LEVEL_MAX_VAL - 1)).astype(np.uint8), GREY_LEVEL_MAX_VAL, [0, GREY_LEVEL_MAX_VAL])
    cumulative = np.cumsum(hist_orig)
    normalized_cumulative = (cumulative / cumulative.max()) * (GREY_LEVEL_MAX_VAL - 1)
    if normalized_cumulative.min() != 0 or normalized_cumulative.max() != (GREY_LEVEL_MAX_VAL - 1):
        np.interp(chanel.flatten(), bins[:-1], normalized_cumulative)
    rounded_cumulative = np.uint8(np.round(normalized_cumulative))
    chanel_in_rounded = rounded_cumulative[(chanel*(GREY_LEVEL_MAX_VAL - 1)).astype(np.uint8)]
    im_eq = (chanel_in_rounded / (GREY_LEVEL_MAX_VAL - 1)).astype(np.float64)
    hist_eq = np.histogram(chanel_in_rounded, GREY_LEVEL_MAX_VAL, [0, GREY_LEVEL_MAX_VAL])[0]
    if rgb_scale(im_orig):
        yiq[:, :, 0] = im_eq
        im_eq = yiq2rgb(yiq)
        im_eq = np.clip(im_eq, np.float64(0), np.float64(1))
    return [im_eq, hist_orig, hist_eq]


"""
Computing z - the borders which divide the histograms into segments. z is an array with shape
(n_quant+1,). The first and last elements are 0 and 255 respectively.
also Computing q - the values to which each of the segments’ intensities will map. q is also a one
dimensional array, containing n_quant elements. 
"""
def find_Z_and_Q(histogram, n_quant):
    Z = [0]
    Q = []
    pixels_in_pic = histogram.sum()
    pixels_in_quant = pixels_in_pic / n_quant
    index = 0
    pixel_sum = 0
    for quant in range(n_quant):
        while pixel_sum < pixels_in_quant and index < (GREY_LEVEL_MAX_VAL - 1):
            pixel_sum += histogram[index]
            index += 1
        z_i = index
        Z.append(z_i)
        q_i = (Z[quant] + Z[quant + 1]) / 2
        Q.append(int(q_i))
        pixel_sum -= pixels_in_quant
    return Z, Q


"""
calculates Z by minimizing the total intensities error.
"""
def calculate_Z(Q, n_quants):
    Z = [0]
    for i in range(1, n_quants):
        z_i = int((Q[i - 1] + Q[i]) / 2)
        Z.append(z_i)
    Z.append((GREY_LEVEL_MAX_VAL - 1))
    return Z


"""
calculates Q by minimizing the total intensities error.
"""
def calculate_Q(Z, n_quants, histogram):
    Q = []
    for i in range(n_quants):
        denomitor_sum, numerator_sum = 0, 0
        low_bound = Z[i]
        high_bound = Z[i + 1]
        for g in range(low_bound, high_bound + 1):
            denomitor_sum += g * histogram[g]
            numerator_sum += histogram[g]
        q_i = denomitor_sum / numerator_sum
        Q.append(int(q_i))
    return Q


"""
function that performs optimal quantization of a given grayscale or RGB image.
"""
def quantize (im_orig, n_quant, n_iter):
    error, im_quant, iters = [], 0, 0
    converge = False
    chnl = im_orig
    if rgb_scale(im_orig):
        yiq = rgb2yiq(np.array(im_orig))
        chnl = yiq[:, :, 0]
    chanel = (chnl * (GREY_LEVEL_MAX_VAL - 1)).astype(np.uint8)
    histogram, bins = np.histogram(chanel, GREY_LEVEL_MAX_VAL, [0, GREY_LEVEL_MAX_VAL])
    Z, Q = find_Z_and_Q(histogram, n_quant)

    # true for n_iter iterations or until Z converges.
    while True:
        if iters == n_iter or converge:
            break
        iters += 1
        Q = calculate_Q(Z, n_quant, histogram)
        new_z = calculate_Z(Q, n_quant)
        if new_z == Z:
            converge = True
        Z = new_z
        err = quantization_error(histogram, Z, Q, n_quant)
        error.append(err)

    for i in range(n_quant):
        histogram[Z[i] : Z[i + 1]] = Q[i]
    chanel_in_value = histogram[chanel]
    im_quant = (chanel_in_value / (GREY_LEVEL_MAX_VAL - 1)).astype(np.float64)
    if rgb_scale(im_orig):
        yiq[:, :, 0] = im_quant
        im_quant = yiq2rgb(yiq)

    return [im_quant, error]


"""
finds The total error introduced by quantization.
"""
def quantization_error(givven_histogram, Z, Q, n_quants):
    E_square = 0
    for i in range(n_quants):
        low_bound = Z[i]
        high_bound = Z[i + 1]
        for g in range(low_bound, high_bound + 1):
            square = (g - Q[i]) ** 2
            E_square += (givven_histogram[g] * square)
    return E_square
