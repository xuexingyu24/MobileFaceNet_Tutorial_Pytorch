#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 08:57:15 2019
face alignment

@author: AIRocker
"""

import numpy as np
import cv2
import torch
import sys
sys.path.append("..")

def transformation_from_points(points1, points2):
    points1 = points1.astype(np.float64)
    points2 = points2.astype(np.float64)
    c1 = np.mean(points1, axis=0)
    c2 = np.mean(points2, axis=0)
    points1 -= c1
    points2 -= c2
    s1 = np.std(points1)
    s2 = np.std(points2)
    points1 /= s1
    points2 /= s2
    U, S, Vt = np.linalg.svd(points1.T * points2)
    R = (U * Vt).T
    return np.vstack([np.hstack(((s2 / s1) * R, c2.T - (s2 / s1) * R * c1.T)), np.matrix([0., 0., 1.])])

def Face_alignment(img,default_square = True,landmarks = []):
    # face alignment -- similarity transformation
    faces = []
    if landmarks != []:
        for i in range(landmarks.shape[0]):
            landmark = landmarks[i, :]
            landmark = landmark.reshape(2, 5).T

            if default_square:

                coord5point =  [[38.29459953, 51.69630051],
                                [73.53179932, 51.50139999],
                                [56.02519989, 71.73660278],
                                [41.54930115, 92.3655014 ],
                                [70.72990036, 92.20410156]]

                pts1 = np.float64(np.matrix([[point[0], point[1]] for point in landmark]))
                pts2 = np.float64(np.matrix([[point[0], point[1]] for point in coord5point]))
                M = transformation_from_points(pts1, pts2)
                aligned_image = cv2.warpAffine(img, M[:2], (img.shape[1], img.shape[0]))
                crop_img = aligned_image[0:112, 0:112]
                faces.append(crop_img)

            else:

                coord5point =  [[30.29459953, 51.69630051],
                                [65.53179932, 51.50139999],
                                [48.02519989, 71.73660278],
                                [33.54930115, 92.3655014],
                                [62.72990036, 92.20410156]]

                pts1 = np.float64(np.matrix([[point[0], point[1]] for point in landmarks_one]))
                pts2 = np.float64(np.matrix([[point[0], point[1]] for point in coord5point]))
                M = transformation_from_points(pts1, pts2)
                aligned_image = cv2.warpAffine(img, M[:2], (img.shape[1], img.shape[0]))
                crop_img = aligned_image[0:112, 0:96]
                faces.append(crop_img)

    return faces


