import cv2
import numpy as np

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
    lines = cv2.HoughLinesP(img, rho=1, theta=np.pi / 180, threshold=50, minLineLength=100, maxLineGap=50)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)


def processImage(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    mask = cv2.inRange(img, low_white, up_white)
    return cv2.Canny(mask, 75, 150)


# Video en directo de la webcam
video = cv2.VideoCapture(0)

while True:
    check, frame = video.read()

    processedImage = processImage(frame)

    croppedImage = regionOfInterest(processedImage)
    drawLines(croppedImage)

    cv2.imshow("ROBOT VISION", frame)
    # cv2.imshow("CROPPED IMAGE",croppedImage)
    # cv2.imshow("PROCESSED IMAGE",processedImage)

    if cv2.waitKey(1) == 27:
        break

video.release()
cv2.destroyAllWindows()
