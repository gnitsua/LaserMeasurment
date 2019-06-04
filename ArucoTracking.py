import cv2
import numpy as np
from cv2 import aruco


class ArucoTracker:
    def __init__(self, marker_size=2):
        self.marker_size = marker_size
        self.corners = None
        self.ids = None
        self.aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_100)

    def pixelDistanceToRealMeasurment(self, pixel_distance):
        if(self.corners is not None):
            # print(self.corners[:,0,0])
            # print(self.corners[:,1,0])
            average_height = np.abs(np.mean(self.corners[:,0,0]-self.corners[:,1,0]))
            return pixel_distance*self.marker_size/average_height
        else:
            return pixel_distance #TODO: what should I do here

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

    def straighten(self, img):
        # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        parameters = aruco.DetectorParameters_create()


        raw_corners, ids, rejectedImgPoints = aruco.detectMarkers(img, self.aruco_dict, parameters=parameters)

        img = aruco.drawDetectedMarkers(img, raw_corners)

        # try:
        #     retval, cameraMatrix, distCoeffs, rvecs, tvecs = aruco.calibrateCameraCharuco(raw_corners, ids, self.board,None, None, None)
        #     print(cameraMatrix)
        # except Exception as e:
        #     raise e
        #     pass
        outer_corners = np.array([[0, 0], [1000, 0], [1000, 500], [0, 500]])
        self.corners = np.array(raw_corners).squeeze()

        # try:
        if (ids is not None and ids.shape[0] == 4):
            ordered_object_points = self.order_points(outer_corners).astype(np.int32)
            ordered_image_points = self.order_points(self.corners.reshape((16, 2))).astype(np.int32)
            # img = cv2.polylines(img, [ordered_object_points.reshape((-1, 1, 2))], True, (0, 255, 0))
            img = cv2.polylines(img, [ordered_image_points.reshape((-1, 1, 2))], True, (255, 0, 0))

            ordered_object_points = np.array(ordered_object_points, dtype="float32")
            ordered_image_points = np.array(ordered_image_points, dtype="float32")

            # M = cv2.getPerspectiveTransform(ordered_image_points[:,0],ordered_object_points[:,0])
            M = cv2.getPerspectiveTransform(ordered_image_points, ordered_object_points)
            img = cv2.warpPerspective(img, M, (1000, 500))
            return img
        else:
            print("not enough trackers")
            return img
        # except AttributeError as e:
        #     print(str(e))
        #     return img
