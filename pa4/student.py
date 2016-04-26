# Please place imports here.
# BEGIN IMPORTS

import numpy as np
import cv2
import scipy
from scipy import ndimage

# END IMPORTS


def compute_photometric_stereo_impl(lights, images):
    """
    Given a set of images taken from the same viewpoint and a corresponding set
    of directions for light sources, this function computes the albedo and
    normal map of a Lambertian scene.

    If the computed albedo for a pixel has an L2 norm less than 1e-7, then set
    the albedo to black and set the normal to the 0 vector.

    Normals should be unit vectors.

    Input:
        lights -- 3 x N array.  Columns are normalized and are to be interpreted
                  as lighting directions.
        images -- list of N images.  Each image is of the same scene from the
                  same viewpoint, but under the lighting condition specified in
                  lights.  Images are height x width x channels arrays, all
                  with identical dimensions.
    Output:
        albedo -- float32 height x width x channels image with dimensions
                  matching the input images.
        normals -- float32 height x width x 3 image with dimensions matching
                   the input images.
    """

    img_shape = np.shape(images[0])
    n = len(images)
    channels = img_shape[2]
    I = np.zeros((img_shape[0], img_shape[1], n))
    for i in xrange (n):
        I[:, :, i] = images[i][:, :, 0]
        # for j in xrange(channels):
    	   # I[:, :, i] += images[i][:, :, j]
        # I[:, :, i] /= channels

    G = I.dot(lights.T).dot(np.linalg.inv(lights.dot(lights.T)))
    kd = np.linalg.norm(G, axis=2)
    zeros = np.where(kd == 0)
    kd[zeros] = 1e-7
    normals = np.divide(G, kd[:, :, np.newaxis])
    normals[zeros] = 0

    albedo = np.zeros(img_shape)
    I_currentChannel = np.zeros((img_shape[0], img_shape[1]))
    for i in xrange(channels):
        albedo_num = np.zeros((img_shape[0], img_shape[1]))
        albedo_den = np.zeros((img_shape[0], img_shape[1]))
        for j in xrange(n):
            I_currentChannel = images[j][:, :, i]
            albedo_num += I_currentChannel * normals.dot(lights[:, j])
            albedo_den += np.square(normals.dot(lights[:, j]))
        albedo_den[albedo_den == 0] = 1
        albedo[:, :, i] = np.divide(albedo_num, albedo_den)
    # albedo[np.linalg.norm(albedo) < 1e-7] = 0

    return albedo.astype(np.float32), normals.astype(np.float32)


def pyrdown_impl(image):
    """
    Prefilters an image with a gaussian kernel and then downsamples the result
    by a factor of 2.

    The following 1D convolution kernel should be used in both the x and y
    directions.
    K = 1/16 [ 1 4 6 4 1 ]

    Functions such as cv2.GaussianBlur and
    scipy.ndimage.filters.gaussian_filter are prohibited.  You must implement
    the separable kernel.  However, you may use functions such as cv2.filter2D
    or scipy.ndimage.filters.correlate to do the actual
    correlation / convolution.

    Filtering should mirror the input image across the border.
    For scipy this is mode = mirror.
    For cv2 this is mode = BORDER_REFLECT_101.

    Downsampling should take the even-numbered coordinates with coordinates
    starting at 0.

    Input:
        image -- height x width [x channels] image of type float32.
    Output:
        down -- ceil(height/2) x ceil(width/2) [x channels] image of type
                float32.
    """
    # raise NotImplementedError()
    K = np.array((1.0, 4.0, 6.0, 4.0, 1.0)) / 16.0
    K = K[:, np.newaxis]

    temp = cv2.filter2D(image, -1, K, borderType = cv2.BORDER_REFLECT_101)
    temp = cv2.filter2D(temp, -1, K.T, borderType = cv2.BORDER_REFLECT_101)
    down = temp[::2, ::2, :]
    
    return down.astype(np.float32)


def pyrup_impl(image):
    """
    Upsamples an image by a factor of 2 and then uses a gaussian kernel as a
    reconstruction filter.

    The following 1D convolution kernel should be used in both the x and y
    directions.
    K = 1/8 [ 1 4 6 4 1 ]
    Note: 1/8 is not a mistake.  The additional factor of 4 (applying this 1D
    kernel twice) scales the solution according to the 2x2 upsampling factor.

    Filtering should mirror the input image across the border.
    For scipy this is mode = mirror.
    For cv2 this is mode = BORDER_REFLECT_101.

    Upsampling should produce samples at even-numbered coordinates with
    coordinates starting at 0.

    Input:
        image -- height x width [x channels] image of type float32.
    Output:
        up -- 2 height x 2 width [x channels] image of type float32.
    """

    img_shape = np.shape(image)
    if len(img_shape) == 2:
        new_img = np.zeros([2 * img_shape[0], 2 * img_shape[1], 1])
    else:
        new_img = np.zeros([2 * img_shape[0], 2 * img_shape[1], img_shape[2]])

    for y in xrange(img_shape[0]):
        for x in xrange(img_shape[1]):
            new_img[2 * y, 2 * x] = image[y,x]

    K = np.array((1.0, 4.0, 6.0, 4.0, 1.0)) / 8.0
    K = K[:, np.newaxis]

    new_img = cv2.filter2D(new_img, -1, K, borderType = cv2.BORDER_REFLECT_101)
    new_img = cv2.filter2D(new_img, -1, K.T, borderType = cv2.BORDER_REFLECT_101)

    return new_img.astype(np.float32)


def project_impl(K, Rt, points):
    """
    Project 3D points into a calibrated camera.

    If the point has a depth < 1e-7 from the camera or is located behind the
    camera, then set the projection to [np.nan, np.nan].

    Input:
        K -- camera intrinsics calibration matrix
        Rt -- 3 x 4 camera extrinsics calibration matrix
        points -- height x width x 3 array of 3D points
    Output:
        projections -- height x width x 2 array of 2D projections
    """
    
    (height, width, _) = np.shape(points)

    projections = np.zeros([height, width, 2])

    for y in xrange(height):
        for x in xrange(width):
            point = points[y, x]
            location = np.ones([4])
            location[:3] = point
            location = Rt.dot(location.T)
            location = K.dot(location.T)
            if location[2] < 1e-7:
                projections[y, x] = [np.nan, np.nan]
                continue
            location = location / location[2]
            projections[y, x] = location[:2]

    return projections



def unproject_corners_impl(K, width, height, depth, Rt):
    """
    Undo camera projection given a calibrated camera and the depth for each
    corner of an image.

    The output points array is a 2x2x3 array arranged for these image
    coordinates in this order:

     (0, 0)      |  (width, 0)
    -------------+------------------
     (0, height) |  (width, height)

    Each of these contains the 3 vector for the corner's corresponding
    point in 3D.

    Tutorial:
      Say you would like to unproject the pixel at coordinate (x, y)
      onto a plane at depth z with camera intrinsics K and camera
      extrinsics Rt.

      (1) Convert the coordinates from homogeneous image space pixel
          coordinates (2D) to a local camera direction (3D):
          (x', y', 1) = K^-1 * (x, y, 1)
      (2) This vector can also be interpreted as a point with depth 1 from
          the camera center.  Multiply it by z to get the point at depth z
          from the camera center.
          (z * x', z * y', z) = z * (x', y', 1)
      (3) Use the inverse of the extrinsics matrix, Rt, to move this point
          from the local camera coordinate system to a world space
          coordinate.
          Note:
            | R t |^-1 = | R' -R't |
            | 0 1 |      | 0   1   |

          p = R' * (z * x', z * y', z, 1) - R't

    Input:
        K -- camera intrinsics calibration matrix
        width -- camera width
        height -- camera height
        depth -- depth of plane with respect to camera
        Rt -- 3 x 4 camera extrinsics calibration matrix
    Output:
        points -- 2 x 2 x 3 array of 3D points
    """
    # raise NotImplementedError()
    corners = np.zeros([2, 2, 3])
    corners[0, 0] = [0, 0, 1]
    corners[0, 1] = [width, 0, 1]
    corners[1, 0] = [0, height, 1]
    corners[1, 1] = [width, height, 1]

    for i in xrange (2):
        for j in xrange (2):
            corners[i, j] = np.linalg.inv(K).dot(corners[i, j])
    corners *= depth

    R = np.zeros((4,3))
    t = np.ones((4,1))
    R[:3] = Rt[:, :3]
    t[:3, 0] = Rt[:, 3]

    temp = np.ones([2, 2, 4])
    temp[:, :, :3] = corners
    for i in xrange (2):
        for j in xrange (2):
            corners[i, j] = ((R.T).dot(temp[i, j, np.newaxis].T) - (R.T).dot(t))[:, 0]

    return corners


def preprocess_ncc_impl(image, ncc_size):
    """
    Prepare normalized patch vectors according to normalized cross
    correlation.

    This is a preprocessing step for the NCC pipeline.  It is expected that
    'preprocess_ncc' is called on every input image to preprocess the NCC
    vectors and then 'compute_ncc' is called to compute the dot product
    between these vectors in two images.

    NCC preprocessing has two steps.
    (1) Compute and subtract the mean.
    (2) Normalize the vector.

    The mean is per channel.  i.e. For an RGB image, over the ncc_size**2
    patch, compute the R, G, and B means separately.  The normalization
    is over all channels.  i.e. For an RGB image, after subtracting out the
    RGB mean, compute the norm over the entire (ncc_size**2 * channels)
    vector and divide.

    If the norm of the vector is < 1e-6, then set the entire vector for that
    patch to zero.

    Patches that extend past the boundary of the input image at all should be
    considered zero.  Their entire vector should be set to 0.

    Patches of shape channels x height x width (e.g. 3 x ncc_size x ncc_size)
    are to be flattened into vectors with the default numpy row major order.
    For example, given the following 2 (channels) x 2 (height) x 2 (width)
    patch, here is how the output vector should be arranged.

    channel1         channel2
    +------+------+  +------+------+ height
    | x111 | x121 |  | x112 | x122 |  |
    +------+------+  +------+------+  |
    | x211 | x221 |  | x212 | x222 |  |
    +------+------+  +------+------+  v
    width ------->

    v = [ x111, x121, x211, x221, x112, x122, x212, x222 ]

    Input:
        image -- height x width x channels image of type float32
        ncc_size -- integer width and height of NCC patch region.
    Output:
        normalized -- heigth x width x (channels * ncc_size**2) array
    """
    # raise NotImplementedError()

    img_shape = np.shape(image)

    # normalized = np.zeros([img_shape[0], img_shape[1], img_shape[2] * ncc_size **2])

    # for y in xrange (img_shape[0]):
    #     for x in xrange (img_shape[1]):
    #         temp = np.zeros([img_shape[2], ncc_size, ncc_size])
    #         low = - (ncc_size / 2)
    #         high = ncc_size / 2
    #         if not (y + low < 0 or x + low < 0 or y + high >= img_shape[0] or x + high >= img_shape[1]):
    #             for k in xrange (img_shape[2]):
    #                 temp[k, :, :] = image[y + low : y + high + 1, x + low : x + high + 1, k]
    #                 temp[k, :, :] -= np.mean(temp[k, :, :])
    #         normalized_vec = np.asarray(temp).reshape(-1)
    #         norm = np.linalg.norm(normalized_vec)
    #         if norm != 0:
    #             normalized[y, x, :] = normalized_vec / norm

    # return normalized

    temp = np.zeros([img_shape[0], img_shape[1], img_shape[2] * ncc_size * ncc_size])
    low = - (ncc_size / 2)
    high = ncc_size / 2

    for y in xrange (img_shape[0]):
        for x in xrange (img_shape[1]):
            if not (y + low < 0 or x + low < 0 or y + high >= img_shape[0] or x + high >= img_shape[1]):
                for k in xrange (img_shape[2]):
                    # print np.asarray(image[y + low : y + high + 1, x + low : x + high + 1, k])
                    # print np.shape(temp[y, x, k : (k+1) * ncc_size * ncc_size])
                    temp[y, x, k * ncc_size * ncc_size: (k+1) * ncc_size * ncc_size] = np.asarray(image[y + low : y + high + 1, x + low : x + high + 1, k]).reshape(-1)
    for k in xrange (img_shape[2]):
        temp[:, :, k * ncc_size * ncc_size: (k+1) * ncc_size * ncc_size] -= np.mean(temp[:, :, k * ncc_size * ncc_size: k * ncc_size * ncc_size], axis=2)[:,:,np.newaxis]
    norm = np.linalg.norm(temp, axis=2)[:,:, np.newaxis]
    for y in xrange (img_shape[0]):
        for x in xrange (img_shape[1]):
            print norm
            if norm[y, x, 0] != 0:
                temp[y, x, :] = temp[y, x, :] / norm[y, x, 0]
            # normalized[norm == 0] = temp / norm

    return temp


def compute_ncc_impl(image1, image2):
    """
    Compute normalized cross correlation between two images that already have
    normalized vectors computed for each pixel with preprocess_ncc.

    Input:
        image1 -- height x width x (channels * ncc_size**2) array
        image2 -- height x width x (channels * ncc_size**2) array
    Output:
        ncc -- height x width normalized cross correlation between image1 and
               image2.
    """
    # raise NotImplementedError()

    img_shape = np.shape(image1)
    ncc = np.zeros((img_shape[0], img_shape[1]))
    for y in xrange (img_shape[0]):
        for x in xrange (img_shape[1]):
            ncc[y, x] = image1[y, x].dot(image2[y, x])

    return ncc
