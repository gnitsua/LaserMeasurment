import cv2
import numpy as np


class RedDotTracking():
    def __init__(self):
        pass

    def getRectangle(self):
        return {"center": (int(self.x), int(self.y)), "width": self.w, "height":self.h}


    def findContourCenter(self,contour):
        M = cv2.moments(contour)
        return np.array([int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])])

    def fitSquare(self, img):
        # img = cv2.resize(img, None, fx=0.5, fy=0.5)
        #   plt.imshow(img)
        #   plt.show()

        # split off the red channel
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.blur(hsv, (10, 10))
        mask = cv2.inRange(mask, (150, 150, 100), (180, 255, 255))
        # mask2 = cv2.inRange(mask, (0, 0, 225), (180, 10, 255))

        ## Merge the mask and crop the red regions
        # mask = cv2.bitwise_or(mask1, mask2)

        # kernel = np.ones((2, 2), np.uint8)
        # mask = cv2.erode(mask, kernel, iterations=2)
        # mask = cv2.dilate(mask, kernel, iterations=2)

        img[np.where(mask == 0)] = 0
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0]
        cnts = list(filter(lambda contour: len(contour) > 10, cnts))

        # only proceed if at least one contour was found
        if len(cnts) > 0:
            # find the largest contour in the mask, then use
            # it to compute the minimum enclosing circle and
            # centroid
            # print(img.shape[0]//2)
            center = np.array([img.shape[1]//2,img.shape[0]//2])
            cv2.circle(img, (center[0],center[1]), 5, (0, 0, 255), -1)


            c = min(cnts, key=lambda kp: np.linalg.norm(self.findContourCenter(kp) - center))
            # c = max(cnts, key=cv2.contourArea)
            # (x, y), (MA, ma), angle = cv2.fitEllipse(c)
            c = cv2.convexHull(c)
            epsilon = 0.001 * cv2.arcLength(c, True)
            c = cv2.approxPolyDP(c, epsilon, True)
            x, y, w, h = cv2.boundingRect(c)
            img = cv2.polylines(img,[c], True, (0,255, 0))
            #       M = cv2.moments(c)
            #       center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            # only proceed if the radius meets a minimum size
            if w > 10:
                # draw the circle and centroid on the frame,
                # then update the list of tracked points
                # cv2.ellipse(img, (int(x), int(y)), (int(MA), int(ma)), int(angle), 0, 360, color=(0, 255, 255))
                cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
                self.x = x
                self.y = y
                self.w = w
                self.h = h

                #           cv2.circle(img, center, 5, (0, 0, 255), -1)

                return img
        return img
