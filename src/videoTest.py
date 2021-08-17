import os.path

import cv2
import numpy as np

rootPath = os.path.abspath(os.path.dirname(__file__))
videoPath = os.path.join(rootPath, "../tests/testWhiteRight.mp4")

# Filter colors
low_white = np.array([5, 5, 160])
up_white = np.array([179, 85, 235])
low_yellow = np.array([18, 94, 140])
up_yellow = np.array([48, 255, 255])


def regionOfInterest(img):
    height = frame.shape[0]
    width = frame.shape[1]
    triangle = [np.array([(150, height), (850, height), (450, 320)])]
    trapeze = [np.array([(0, height), (width / 2, height / 2), (width, height)], dtype=np.int32)]
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, trapeze, 255)
    return cv2.bitwise_and(img, mask)


def drawLines(img):
    lines = cv2.HoughLinesP(img, rho=1, theta=np.pi/180, threshold=50, minLineLength=70, maxLineGap=20)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)


def processImage(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    mask = cv2.inRange(img, low_white, up_white)
    return cv2.Canny(mask, 75, 150)


# Video pregrabado
video = cv2.VideoCapture(videoPath)

pos_frame = video.get(cv2.CAP_PROP_POS_FRAMES)

while True:
    flag, frame = video.read()

    # Loop video
    if not flag:
        video = cv2.VideoCapture(videoPath)
        continue

    processedImage = processImage(frame)

    croppedImage = regionOfInterest(processedImage)
    drawLines(croppedImage)

    cv2.imshow("Video pregrabado", frame)
    cv2.imshow("Cropped image", croppedImage)
    pos_frame = video.get(cv2.CAP_PROP_POS_FRAMES)
    # print(str(pos_frame) + " frames y " + str(video.get(cv2.CAP_PROP_POS_MSEC)/1000) + " segundos")

    if cv2.waitKey(10) == 27:
        break

video.release()
cv2.destroyAllWindows()
