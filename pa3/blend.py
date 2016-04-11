import math
import sys

import cv2
import numpy as np


class ImageInfo:
    def __init__(self, name, img, position):
        self.name = name
        self.img = img
        self.position = position


def imageBoundingBox(img, M):
    """
       This is a useful helper function that you might choose to implement
       that takes an image, and a transform, and computes the bounding box
       of the transformed image.

       INPUT:
         img: image to get the bounding box of
         M: the transformation to apply to the img
       OUTPUT:
         minX: int for the minimum X value of a corner
         minY: int for the minimum Y value of a corner
         minX: int for the maximum X value of a corner
         minY: int for the maximum Y value of a corner
    """
    #TODO 8
    #TODO-BLOCK-BEGIN
    
    img_shape = img.shape
    vec_1 = np.array((0, 0, 1))
    vec_2 = np.array((img_shape[1], 0, 1))
    vec_3 = np.array((0, img_shape[0], 1))
    vec_4 = np.array((img_shape[1], img_shape[0], 1))

    vec_1trans = M.dot(vec_1)
    vec_2trans = M.dot(vec_2)
    vec_3trans = M.dot(vec_3)
    vec_4trans = M.dot(vec_4)

    vec_1trans /= vec_1trans[2]
    vec_2trans /= vec_2trans[2]
    vec_3trans /= vec_3trans[2]
    vec_4trans /= vec_4trans[2]

    minX = min(vec_1trans[0], vec_2trans[0], vec_3trans[0], vec_4trans[0])
    minY = min(vec_1trans[1], vec_2trans[1], vec_3trans[1], vec_4trans[1])
    maxX = max(vec_1trans[0], vec_2trans[0], vec_3trans[0], vec_4trans[0])
    maxY = max(vec_1trans[1], vec_2trans[1], vec_3trans[1], vec_4trans[1])

    #TODO-BLOCK-END
    return int(minX), int(minY), int(maxX), int(maxY)


def accumulateBlend(img, acc, M, blendWidth):
    """
       INPUT:
         img: image to add to the accumulator
         acc: portion of the accumulated image where img should be added
         M: the transformation mapping the input image to the accumulator
         blendWidth: width of blending function. horizontal hat function
       OUTPUT:
         modify acc with weighted copy of img added where the first
         three channels of acc record the weighted sum of the pixel colors
         and the fourth channel of acc records a sum of the weights
    """
    # BEGIN TODO 10
    # Fill in this routine
    #TODO-BLOCK-BEGIN
    
    acc_shape = acc.shape
    img_shape = img.shape
    minX, minY, maxX, maxY = imageBoundingBox(img, M)
    M_inverse = np.linalg.inv(M)
    for y in xrange(minY, maxY):
        for x in xrange(minX, maxX):
            vec = np.array((float(x), float(y), 1.0))
            vec_inverse = M_inverse.dot(vec)
            vec_inverse /= vec_inverse[2]
            if not(vec_inverse[0] < 0 or vec_inverse[0] >= img_shape[1] - 1 or vec_inverse[1] < 0 or vec_inverse[1] >= img_shape[0] - 1):
                lowX = np.floor(vec_inverse[0])
                lowY = np.floor(vec_inverse[1])
                highX = lowX + 1
                highY = lowY + 1
                a = vec_inverse[0] - lowX
                b = vec_inverse[1] - lowY
                ul = img[lowY, lowX] # upper left neighbor
                ur = img[lowY, highX]
                bl = img[highY, lowX]
                br = img[highY, highX]

                if np.sum(ul[0:3]) != 0 and np.sum(ur[0:3]) != 0 and np.sum(bl[0:3]) != 0 and np.sum(br[0:3]) != 0:

                    pixel = (1 - b) * ((1 - a) * ul + a * ur) + b * ((1 - a) * bl + a * br)

                    weight = 0.0
                    weightx = 0.0
                    weighty = 0.0
                    if vec_inverse[0] < blendWidth:
                        weightx = vec_inverse[0] / blendWidth
                    elif vec_inverse[0] > img_shape[1] - blendWidth:
                        weightx = (img_shape[1] - vec_inverse[0]) / blendWidth
                    else:
                        weightx = 1.0

                    if vec_inverse[1] < blendWidth:
                        weighty = vec_inverse[1] / blendWidth
                    elif vec_inverse[1] > img_shape[0] - blendWidth:
                        weighty = (img_shape[0] - vec_inverse[1]) / blendWidth
                    else:
                        weighty = 1.0

                    weight = min(weightx, weighty)

                    acc[y, x, 0] += pixel[0] * weight
                    acc[y, x, 1] += pixel[1] * weight
                    acc[y, x, 2] += pixel[2] * weight
                    acc[y, x, 3] += weight
    #TODO-BLOCK-END
    # END TODO


def normalizeBlend(acc):
    """
       INPUT:
         acc: input image whose alpha channel (4th channel) contains
         normalizing weight values
       OUTPUT:
         img: image with r,g,b values of acc normalized
    """
    # BEGIN TODO 11
    # fill in this routine..
    #TODO-BLOCK-BEGIN
    
    acc_shape = acc.shape
    img = np.zeros((acc_shape), dtype=np.uint8)

    for y in xrange(acc_shape[0]):
        for x in xrange(acc_shape[1]):
            if acc[y, x, 3] != 0:
                img[y, x, :3] = acc[y, x, :3] / acc[y, x, 3]
            img[y, x, 3] = 255

    #TODO-BLOCK-END
    # END TODO
    return img


def blendImages(ipv, blendWidth, is360=False, A_out=None):
    """
       INPUT:
         ipv: list of input images and their relative positions in the mosaic
         blendWidth: width of the blending function
       OUTPUT:
         croppedImage: final mosaic created by blending all images and
         correcting for any vertical drift
    """
    # Compute bounding box for the mosaic
    minX = sys.maxint
    minY = sys.maxint
    maxX = 0
    maxY = 0
    channels = -1
    width = -1  # Assumes all images are the same width
    M = np.identity(3)
    for i in ipv:
        M = i.position
        img = i.img
        _, w, c = img.shape
        if channels == -1:
            channels = c
            width = w

        # BEGIN TODO 9
        # add some code here to update minX, ..., maxY
        #TODO-BLOCK-BEGIN
        
        bounds = imageBoundingBox(img, M)
        minX = min(minX, bounds[0])
        minY = min(minY, bounds[1])
        maxX = max(maxX, bounds[2])
        maxY = max(maxY, bounds[3])

        #TODO-BLOCK-END
        # END TODO

    # Create an accumulator image
    accWidth = int(math.ceil(maxX) - math.floor(minX))
    accHeight = int(math.ceil(maxY) - math.floor(minY))
    print 'accWidth, accHeight:', (accWidth, accHeight)
    acc = np.zeros((accHeight, accWidth, channels + 1))
    # Add in all the images
    translation = np.array([[1, 0, -minX], [0, 1, -minY], [0, 0, 1]])
    M = np.identity(3)
    for count, i in enumerate(ipv):
        M = i.position
        img = i.img

        M_trans = translation.dot(M)
        accumulateBlend(img, acc, M_trans, blendWidth)

        # First image
        if count == 0:
            p = np.array([0.5 * width, 0, 1])
            p = M_trans.dot(p)
            x_init, y_init = p[:2] / p[2]
        # Last image
        if count == (len(ipv) - 1):
            p = np.array([0.5 * width, 0, 1])
            p = M_trans.dot(p)
            x_final, y_final = p[:2] / p[2]
    # print acc
    # print "accumulateBlend done"
    # compImage = acc[:3].astype(np.uint8)
    compImage = normalizeBlend(acc)
    # print compImage
    # print "normalizeBlend done"

    # Determine the final image width
    outputWidth = (accWidth - width) if is360 else accWidth
    # Compute the affine transform
    A = np.identity(3)
    # BEGIN TODO 12
    # fill in appropriate entries in A to trim the left edge and
    # to take out the vertical drift if this is a 360 panorama
    # (i.e. is360 is true)
    # Shift it left by the correct amount
    # Then handle the vertical drift
    # Note: warpPerspective does inverse mapping which means A is an affine
    # transform that maps final panorama coordinates to accumulator coordinates
    #TODO-BLOCK-BEGIN
    
    if is360:
        A[1, 0] = -float(y_init - y_final) / outputWidth
        A[0, 2] = -width / 2

    #TODO-BLOCK-END
    # END TODO

    if A_out is not None:
        A_out[:] = A

    # Warp and crop the composite
    croppedImage = cv2.warpPerspective(
        compImage, A, (outputWidth, accHeight), flags=cv2.INTER_LINEAR
    )

    return croppedImage

