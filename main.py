import cv2
import matplotlib.pyplot as plt
import numpy as np
import time
from ArucoTracking import ArucoTracker
from RedDotTracking import RedDotTracking

if __name__ == "__main__":
    # cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture('2F_300_500_60.webm')
    # cap = cv2.VideoCapture('IMG_1411.TRIM.MOV')
    straightener = ArucoTracker()
    tracker = RedDotTracking()

    heights = []
    _, f = cap.read()
    avg1 = np.float32(f)
    while (True):
        ret, img = cap.read()

        if (ret == True):

            cv2.accumulateWeighted(img, avg1, 0.5)

            img = cv2.convertScaleAbs(img)

            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # retval, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(objectPoints, imagePoints)
            img = straightener.straighten(img)
            img = tracker.fitSquare(img)

            rectangle = tracker.getRectangle()
            heights.append(straightener.pixelDistanceToRealMeasurment(rectangle["height"]))
            cv2.imshow('frame', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

    plt.plot(np.linspace(300, 500, len(heights)), heights);
    plt.title("Vertical Displacement")
    plt.xlabel("Hz")
    plt.ylabel("Displacement (inches)")
    plt.savefig('2F_300_500_60_%d.png'%time.time())
    plt.show()
