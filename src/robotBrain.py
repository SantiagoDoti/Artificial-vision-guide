import cv2, time, numpy as np

# Colores blancos para filtrar
low_white = np.array([5,5,160])
up_white = np.array([179,85,235])

def interestRegion(img, vertices):
    mask = np.zeros_like(img)
    ignoreMaskColor = 255
    cv2.fillPoly(mask,vertices,ignoreMaskColor)
    maskedImage = cv2.bitwise_and(img,mask)
    return maskedImage

def drawLines(img):
    lines = cv2.HoughLinesP(img,1, np.pi/180,50, maxLineGap=50)
    if lines is not None:
        for line in lines:
            x1,y1,x2,y2 = line[0]
            cv2.line(frame,(x1,y1),(x2,y2),(0,255,0),5)

def processImage(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    img = cv2.GaussianBlur(img,(5,5),0)
    mask = cv2.inRange(img,low_white,up_white)
    finalImage = cv2.Canny(mask,75,150)
    return finalImage


video = cv2.VideoCapture(0)

while True:

    check, frame = video.read()

    processedImage = processImage(frame)
        
    # Vertices de un trapecio
    verticesRegionInterest = [np.array([[0,480],[0,300],[220,200],[420,200],[640,300],[640,480]],dtype=np.int32)]
    croppedImage = interestRegion(processedImage,verticesRegionInterest)
    drawLines(croppedImage)

    cv2.imshow("ROBOT VISION",frame)
    cv2.imshow("CROPPED IMAGE",croppedImage)
    cv2.imshow("PROCESSED IMAGE",processedImage)

    key = cv2.waitKey(1)
    if key == 27:
        break

video.release()
cv2.destroyAllWindows()


    