import cv2
import numpy as np

def cross_correlation_2d(img, kernel):
    '''Given a kernel of arbitrary m x n dimensions, with both m and n being
    odd, compute the cross correlation of the given image with the given
    kernel, such that the output is of the same dimensions as the image and that
    you assume the pixels out of the bounds of the image to be zero. Note that
    you need to apply the kernel to each channel separately, if the given image
    is an RGB image.

    Inputs:
        img:    Either an RGB image (height x width x 3) or a grayscale image
                (height x width) as a numpy array.
        kernel: A 2D numpy array (m x n), with m and n both odd (but may not be
                equal).

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    # TODO-BLOCK-BEGIN
    dim = len(img.shape)
    kernelDim = kernel.shape
    def getImg(img2D):
        ret2 = np.empty(img2D.shape)
        for y in xrange(img2D.shape[0]):
            for x in xrange(img2D.shape[1]):
                val = 0
                for m in xrange(kernelDim[0]/-2 + 1, kernelDim[0]/2 + 1):
                    for n in xrange(kernelDim[1]/-2 + 1, kernelDim[1]/2 + 1):
                        if (y + m < 0 or y + m >= img2D.shape[0] or 
                            x + n < 0 or x + n >= img2D.shape[1]):
                            continue
                        else:
                            val += (img2D[y + m][x + n] * 
                                kernel[m + kernelDim[0]/2][n + kernelDim[1]/2])
                ret2[y][x] = val
        return ret2
    if dim == 2:
        return getImg(img)
    if dim == 3:
        ret = np.empty(img.shape)
        ret[:,:,0] = getImg(img[:,:,0])
        ret[:,:,1] = getImg(img[:,:,1])
        ret[:,:,2] = getImg(img[:,:,2])
        return ret
    # TODO-BLOCK-END

def convolve_2d(img, kernel):
    '''Use cross_correlation_2d() to carry out a 2D convolution.

    Inputs:
        img:    Either an RGB image (height x width x 3) or a grayscale image
                (height x width) as a numpy array.
        kernel: A 2D numpy array (m x n), with m and n both odd (but may not be
                equal).

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    # TODO-BLOCK-BEGIN
    return cross_correlation_2d(img, np.flipud(np.fliplr(kernel)))
    # TODO-BLOCK-END

def gaussian_blur_kernel_2d(sigma, width, height):
    '''Return a Gaussian blur kernel of the given dimensions and with the given
    sigma. Note that width and height are different.

    Input:
        sigma:  The parameter that controls the radius of the Gaussian blur.
                Note that, in our case, it is a circular Gaussian (symmetric
                across height and width).
        width:  The width of the kernel.
        height: The height of the kernel.

    Output:
        Return a kernel of dimensions width x height such that convolving it
        with an image results in a Gaussian-blurred image.
    '''
    # TODO-BLOCK-BEGIN
    ret = np.empty([height, width])
    kernelSum = 0.0
    for y in xrange(height):
        for x in xrange(width):
            ret[y,x] = 1./2/np.pi/sigma**2 * np.exp(-((y-(height/2))**2 
                + (x-(width/2))**2)/2/sigma**2)
            kernelSum += ret[y,x]
    # ratio = 1.0/np.sum(ret)
    # print np.sum(ret)
    ratio = 1.0/kernelSum
    print kernelSum
    print ratio
    for y in xrange(height):
    	for x in xrange(width):
    		ret[y,x] = ret[y,x] * ratio
    # print ret
    return ret
    # TODO-BLOCK-END

def low_pass(img, sigma, size):
    '''Filter the image as if its filtered with a low pass filter of the given
    sigma and a square kernel of the given size. A low pass filter supresses
    the higher frequency components (finer details) of the image.

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    # TODO-BLOCK-BEGIN
    kernel = gaussian_blur_kernel_2d(sigma, size, size)
    return convolve_2d(img, kernel)
    # TODO-BLOCK-END

def high_pass(img, sigma, size):
    '''Filter the image as if its filtered with a high pass filter of the given
    sigma and a square kernel of the given size. A high pass filter suppresses
    the lower frequency components (coarse details) of the image.

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    # TODO-BLOCK-BEGIN
    return img - low_pass(img, sigma, size)
    # TODO-BLOCK-END

def create_hybrid_image(img1, img2, sigma1, size1, high_low1, sigma2, size2,
        high_low2, mixin_ratio):
    '''This function adds two images to create a hybrid image, based on
    parameters specified by the user.'''
    high_low1 = high_low1.lower()
    high_low2 = high_low2.lower()

    if img1.dtype == np.uint8:
        img1 = img1.astype(np.float32) / 255.0
        img2 = img2.astype(np.float32) / 255.0

    if high_low1 == 'low':
        img1 = low_pass(img1, sigma1, size1)
    else:
        img1 = high_pass(img1, sigma1, size1)

    if high_low2 == 'low':
        img2 = low_pass(img2, sigma2, size2)
    else:
        img2 = high_pass(img2, sigma2, size2)

    img1 *= (1 - mixin_ratio)
    img2 *= mixin_ratio
    hybrid_img = (img1 + img2)
    if len(hybrid_img.shape) == 3: # if its an RGB image
        for c in range(3):
            hybrid_img[:, :, c]  /= np.amax(hybrid_img[:, :, c])
    else:
        hybrid_img /= np.amax(hybrid_img)
    return (hybrid_img * 255).clip(0, 255).astype(np.uint8)


