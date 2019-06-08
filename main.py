import argparse
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np

from ArucoTracking import ArucoTracker
from RedDotTracking import RedDotTracking

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Measurment utility for measuring beam with of a red laser')
    parser.add_argument('--input', type=str, default=0,
                        help='Input video file to process (default uses your webcam)')
    parser.add_argument('--calibration', type=str, default=None,
                        help='Calibration file for the camera (using opencv)')
    parser.add_argument('--name', type=str, default="Test %d" % time.time(),
                        help='Test name, used for naming output files')

    args = parser.parse_args()

    cap = cv2.VideoCapture(args.input)
    # cap = cv2.VideoCapture('2F_300_500_60.webm')
    # cap = cv2.VideoCapture('calibration_video/6.webm')
    # cap = cv2.VideoCapture('IMG_1411.TRIM.MOV')
    if (args.calibration is not None):
        npzfile = np.load(args.calibration)
        h = npzfile['h']
        w = npzfile['w']
        mtx = npzfile['mtx']
        dist = npzfile['dist']
        straightener = ArucoTracker(mtx, dist)
    else:
        straightener = ArucoTracker()
    tracker = RedDotTracking()

    heights = []
    widths = []
    xs = []
    ys = []
    _, f = cap.read()
    avg1 = np.float32(f)
    print("Processing started, press q to quit")
    while (True):

        ret, img = cap.read()

        if (ret == True):

            cv2.accumulateWeighted(img, avg1, 0.5)

            img = cv2.convertScaleAbs(img)
            img = straightener.straighten(img)
            img = tracker.fitSquare(img)

            rectangle = tracker.getRectangle()
            heights.append(straightener.pixelDistanceToRealMeasurment(rectangle["height"]))
            widths.append(straightener.pixelDistanceToRealMeasurment(rectangle["width"]))
            xs.append(straightener.pixelDistanceToRealMeasurment(rectangle["height"]))
            ys.append(straightener.pixelDistanceToRealMeasurment(rectangle["width"]))
            cv2.imshow('frame', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
    xs = np.array(xs)
    ys = np.array(ys)
    print("X: %f (%f), Y: %f (%f)" % (np.mean(xs), np.std(xs), np.mean(ys), np.std(ys)))

    plt.plot(np.linspace(300, 500, len(heights)), heights);
    # plt.plot(np.linspace(300, 500, len(heights)), widths);
    plt.title("2F")
    plt.xlabel("Hz")
    plt.ylabel("Displacement (inches)")
    # plt.legend(["Vertical Displacement","Horizontal Displacement"])
    plt.savefig('2F_300_500_60_%d.png' % time.time())
    plt.show()
