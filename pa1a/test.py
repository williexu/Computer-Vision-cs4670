"""
Convention: append an integer to the end of the test, for multiple versions of
the same test at different difficulties.  Higher numbers are more difficult
(lower thresholds or accept fewer mistakes).  Example:
    test_all_equal1(self):
        ...
    test_all_equal2(self):
        ...
"""

import argparse
import json
import os
import math
import random
import unittest

import cv2
import numpy as np

import hybrid

class TestCrossCorrelation2D(unittest.TestCase):
    def setUp(self):
        self.small_height = 10
        self.small_width = 8
        self.big_height = 50
        self.big_width = 40
        self.big_img_grey = np.random.rand(self.big_height,self.big_width)
        self.small_img_grey = np.random.rand(self.small_height,self.small_width)
        self.img_rgb = np.random.rand(self.big_height,self.big_width,3)

    def tearDown(self):
        pass

    def test_identity_filter_grey(self):
        '''
        Tests whether the cross-correlation identity returns the original image
        '''
        identity = np.zeros((3,3))
        identity[1,1] = 1
        img_dup = hybrid.cross_correlation_2d(self.small_img_grey, identity)
        self.assertTrue(np.allclose(img_dup, self.small_img_grey, atol=1e-08), \
                msg="Failed to return original image under identity cross-correlation")

    def test_mean_filter_grey(self):
        '''
        Tests cross-correlation of greyscale image using mean filter
        '''
        mean = np.ones((3,3))
        student = hybrid.cross_correlation_2d(self.small_img_grey, mean)
        solution = cv2.filter2D(self.small_img_grey, -1, mean, borderType=cv2.BORDER_CONSTANT)

        self.assertTrue(np.allclose(student, solution, atol=1e-08), \
                msg="Incorrect cross-correlation of greyscale image using mean filter")

    def test_mean_filter_rect_grey(self):
        '''
        Tests cross-correlation of greyscale image using a rectangular mean filter
        '''
        mean = np.ones((3,5))
        student = hybrid.cross_correlation_2d(self.small_img_grey, mean)
        solution = cv2.filter2D(self.small_img_grey, -1, mean, borderType=cv2.BORDER_CONSTANT)

        self.assertTrue(np.allclose(student, solution, atol=1e-08), \
                msg="Incorrect cross-correlation of greyscale image using rectangular mean filter")

    def test_mean_filter_RGB(self):
        '''
        Tests cross-correlation of RGB image using a rectangular filter
        '''
        mean = np.ones((3,3))
        student = hybrid.cross_correlation_2d(self.img_rgb, mean)
        solution = cv2.filter2D(self.img_rgb, -1, mean, borderType=cv2.BORDER_CONSTANT)

        self.assertTrue(np.allclose(student, solution, atol=1e-08), \
                msg="Incorrect cross-correlation of RGB image using mean filter")

    def test_rand_rect_filter_RGB(self):
        '''
        Tests cross-correlation of RGB image using a random rectangular filter
        '''
        rand_filt = np.random.rand(5,7)
        student = hybrid.cross_correlation_2d(self.img_rgb, rand_filt)
        solution = cv2.filter2D(self.img_rgb, -1, rand_filt, borderType=cv2.BORDER_CONSTANT)

        self.assertTrue(np.allclose(student, solution, atol=1e-08), \
                msg="Incorrect cross-correlation of RGB image using random rectangular filter")

    def test_big_filter_grey(self):
        '''
        Tests cross-correlation of greyscale image using a filter bigger than image
        '''
        filter_height = self.small_height % 2 + self.small_height + 1
        filter_width = self.small_width % 2 + self.small_width + 1
        rand_filter = np.random.rand(filter_height, filter_width)
        student = hybrid.cross_correlation_2d(self.small_img_grey, rand_filter)
        solution = cv2.filter2D(self.small_img_grey, -1, rand_filter, borderType=cv2.BORDER_CONSTANT)

        self.assertTrue(np.allclose(student, solution, atol=1e-08), \
                msg="Incorrect cross-correlation of greyscale image using filter bigger than image")

class TestConvolve2D(unittest.TestCase):
    def setUp(self):
        self.small_height = 10
        self.small_width = 8
        self.big_height = 50
        self.big_width = 40
        self.big_img_grey = np.random.rand(self.big_height,self.big_width)
        self.small_img_grey = np.random.rand(self.small_height,self.small_width)
        self.img_rgb = np.random.rand(self.big_height,self.big_width,3)

    def tearDown(self):
        pass

    def test_identity_filter_grey(self):
        '''
        Tests whether the convolution identity returns the original image
        '''
        identity = np.zeros((3,3))
        identity[1,1] = 1
        img_dup = hybrid.convolve_2d(self.small_img_grey, identity)
        self.assertTrue(np.allclose(img_dup, self.small_img_grey, atol=1e-08), \
                msg="Failed to return original image under identity convolution")

    def test_mean_filter_grey(self):
        '''
        Tests convolution of greyscale image using mean filter
        '''
        mean = np.ones((3,3))
        student = hybrid.convolve_2d(self.small_img_grey, mean)
        solution = cv2.filter2D(self.small_img_grey, -1, mean, borderType=cv2.BORDER_CONSTANT)

        self.assertTrue(np.allclose(student, solution, atol=1e-08), \
                msg="Incorrect result convolving greyscale image using mean filter")

    def test_mean_filter_rect_grey(self):
        '''
        Tests convolution of greyscale image using a rectangular mean filter
        '''
        mean = np.ones((3,5))
        mean_trans = np.fliplr(np.flipud(mean))
        student = hybrid.convolve_2d(self.small_img_grey, mean)
        solution = cv2.filter2D(self.small_img_grey, -1, mean_trans, borderType=cv2.BORDER_CONSTANT)

        self.assertTrue(np.allclose(student, solution, atol=1e-08), \
                msg="Incorrect result convolving greyscale image using rectangular mean filter")

    def test_mean_filter_RGB(self):
        '''
        Tests convolution of RGB image using a rectangular filter
        '''
        mean = np.ones((3,3))
        student = hybrid.convolve_2d(self.img_rgb, mean)
        solution = cv2.filter2D(self.img_rgb, -1, mean, borderType=cv2.BORDER_CONSTANT)

        self.assertTrue(np.allclose(student, solution, atol=1e-08), \
                msg="Incorrect result convolving RGB image using mean filter")

    def test_rand_rect_filter_RGB(self):
        '''
        Tests convolution of RGB image using a random rectangular filter
        '''
        rand_filt = np.random.rand(5,7)
        rand_filt_trans = np.fliplr(np.flipud(rand_filt))
        student = hybrid.convolve_2d(self.img_rgb, rand_filt)
        solution = cv2.filter2D(self.img_rgb, -1, rand_filt_trans, borderType=cv2.BORDER_CONSTANT)

        self.assertTrue(np.allclose(student, solution, atol=1e-08), \
                msg="Incorrect result convolving RGB image using random rectangular filter")

    def test_big_filter_grey(self):
        '''
        Tests convolution of greyscale image using a filter bigger than image
        '''
        filter_height = self.small_height % 2 + self.small_height + 1
        filter_width = self.small_width % 2 + self.small_width + 1
        rand_filt = np.random.rand(filter_height, filter_width)
        rand_filt_trans = np.fliplr(np.flipud(rand_filt))
        student = hybrid.convolve_2d(self.small_img_grey, rand_filt)
        solution = cv2.filter2D(self.small_img_grey, -1, rand_filt_trans, borderType=cv2.BORDER_CONSTANT)

        self.assertTrue(np.allclose(student, solution, atol=1e-08), \
                msg="Incorrect result convolving greyscale image using filter bigger than image")

if __name__ == '__main__':
    unittest.main()
