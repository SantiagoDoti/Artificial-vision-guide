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
    lines = cv2.HoughLinesP(img,rho=1, theta=np.pi/180,threshold=80, maxLineGap=50)
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

# Video pregrabado
video = cv2.VideoCapture("./Artificial-vision-guide/tests/testWhiteRight.mp4")

pos_frame = video.get(cv2.CAP_PROP_POS_FRAMES)
while True:
    flag, frame = video.read()
    if flag:
        processedImage = processImage(frame)
        # Vertices de un trapecio (region que nos interesa)
        verticesRegionInterest = [np.array([[0,480],[0,300],[220,300],[420,300],[640,300],[640,480]],dtype=np.int32)]
        croppedImage = interestRegion(processedImage,verticesRegionInterest)
        drawLines(croppedImage)

        cv2.imshow("Video pregrabado",frame)
        pos_frame = video.get(cv2.CAP_PROP_POS_FRAMES)
        print(str(pos_frame)+ " frames y " +str(video.get(cv2.CAP_PROP_POS_MSEC)/1000) +" segundos")
    else:
        # Reinicio video para que se reproduzca en loop
        print("<---------------------- REINICIANDO VIDEO ---------------------->")
        video.set(cv2.CAP_PROP_POS_FRAMES,0)
    if cv2.waitKey(10) == 27:
        break
    if video.get(cv2.CAP_PROP_POS_FRAMES) == video.get(cv2.CAP_PROP_FRAME_COUNT):
        break

cv2.destroyAllWindows()