import cv2
import numpy as np
from cv2 import aruco


class ArucoTracker:
    def __init__(self, mtx=None, dist=None, marker_size=2):
        self.mtx = mtx
        self.dist = dist
        # self.image_size_x = 1500
        self.image_size_x = 750
        # self.image_size_y = 800
        self.image_size_y = 400
        self.marker_size = marker_size
        self.marker_size_pixels = marker_size * 300  # assuming 300 dpi
        self.corners = None
        self.object_corners = np.array([[[0, 0], [self.marker_size_pixels, 0],
                                         [self.marker_size_pixels, self.marker_size_pixels],
                                         [0, self.marker_size_pixels]],
                                        [[self.image_size_x - self.marker_size_pixels, 0], [self.image_size_x, 0],
                                         [self.image_size_x, self.marker_size_pixels],
                                         [self.image_size_x - self.marker_size_pixels, self.marker_size_pixels]],
                                        [[self.image_size_x - self.marker_size_pixels,
                                          self.image_size_y - self.marker_size_pixels],
                                         [self.image_size_x, self.image_size_y - self.marker_size_pixels],
                                         [self.image_size_x, self.image_size_y],
                                         [self.image_size_x - self.marker_size_pixels, self.image_size_y]],
                                        [[0, self.image_size_y - self.marker_size_pixels],
                                         [self.marker_size_pixels, self.image_size_y - self.marker_size_pixels],
                                         [self.marker_size_pixels, self.image_size_y], [0, self.image_size_y]]],
                                       dtype="float32")
        self.ids = None
        self.aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_100)

    def pixelDistanceToRealMeasurment(self, pixel_distance):
        if (self.corners is not None):
            average_marker_height = np.mean(np.linalg.norm(self.corners[:, 1] - self.corners[:, 0], axis=1))
            average_marker_width = np.mean(np.linalg.norm(self.corners[:, 3] - self.corners[:, 0], axis=1))
            average_marker_size = (average_marker_width + average_marker_height) / 2
            return pixel_distance * self.marker_size / average_marker_size
        else:
            return pixel_distance  # TODO: what should I do here

    def order_points(self, pts):
        # initialzie a list of coordinates that will be ordered
        # such that the first entry in the list is the top-left,
        # the second entry is the top-right, the third is the
        # bottom-right, and the fourth is the bottom-left
        rect = np.zeros((4, 2), dtype="float32")

        # the top-left point will have the smallest sum, whereas
        # the bottom-right point will have the largest sum
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        # now, compute the difference between the points, the
        # top-right point will have the smallest difference,
        # whereas the bottom-left will have the largest difference
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        # return the ordered coordinates
        return rect

    def undistort(self, img):
        newCameraMtx, roi = cv2.getOptimalNewCameraMatrix(self.mtx, self.dist, (self.image_size_x, self.image_size_y), 1, (self.image_size_x, self.image_size_y))
        return cv2.undistort(img, self.mtx, self.dist, None, newCameraMtx)

    def straighten(self, img):

        if self.mtx is not None:#TODO: only checking one
            img = self.undistort(img)
        # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        parameters = aruco.DetectorParameters_create()

        raw_corners, ids, rejectedImgPoints = aruco.detectMarkers(img, self.aruco_dict, parameters=parameters)

        img = aruco.drawDetectedMarkers(img, raw_corners)

        clean_corners = np.array(raw_corners, dtype="float32").squeeze()

        # try:
        if (ids is not None and ids.shape[0] == 4):
            ordered_object_points = self.order_points(self.object_corners.reshape((16, 2))).astype(np.int32)
            ordered_image_points = self.order_points(clean_corners.reshape((16, 2))).astype(np.int32)
            # img = cv2.polylines(img, [ordered_object_points.reshape((-1, 1, 2))], True, (0, 255, 0))
            img = cv2.polylines(img, [ordered_image_points.reshape((-1, 1, 2))], True, (255, 0, 0))

            ordered_object_points = np.array(ordered_object_points, dtype="float32")
            ordered_image_points = np.array(ordered_image_points, dtype="float32")


            # ordered_object_points_3d = np.concatenate((ordered_object_points,np.zeros((ordered_object_points.shape[0],1))),axis=1)
            # ordered_image_points_3d = np.concatenate((ordered_image_points,np.zeros((ordered_object_points.shape[0],1))),axis=1)
            # print(ordered_object_points_3d.shape)
            # retval, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(ordered_object_points_3d,ordered_image_points_3d,(self.image_size_x, self.image_size_y),None,None)
            # img = cv2.fisheye.undistortImage(img, K, D)

            # M = cv2.getPerspectiveTransform(ordered_image_points[:,0],ordered_object_points[:,0])
            M = cv2.getPerspectiveTransform(ordered_image_points, ordered_object_points)
            img = cv2.warpPerspective(img, M, (self.image_size_x, self.image_size_y))
            self.corners = cv2.perspectiveTransform(clean_corners.reshape((1, -1, 2)), M).reshape(4, 4, 2)
            return img
        else:
            print("not enough trackers")
            return img
        # except AttributeError as e:
        #     print(str(e))
        #     return img
