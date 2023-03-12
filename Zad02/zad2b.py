import cv2
import numpy as np

# from ximea import xiapi
# import ipywidgets as widgets
# from IPython.display import display

# param1 = 10
# cam = xiapi.Camera() #use xiapi cam
# cam.open_device()   #open camera
# camera settings
# cam.set_exposure(10000)
# cam.set_param('imgdataformat','XI_RGB32')
# cam.set_param('auto_wb', 1)
# img = xiapi.Image()     #create instance to store image
# cam.start_acquisition() #start data inquisiton

cam = cv2.VideoCapture(0)

iteracia = 0
a = ""

radiusMin = 0
radiusMax = 0
parameter2 = 0.9
minDistance = 10


# create slider
def on_trackbar1(val1):
    global radiusMin
    radiusMin = val1
    print(radiusMin)


def on_trackbar2(val2):
    global radiusMax
    radiusMax = val2
    print(radiusMax)


def on_trackbar3(val3):
    global parameter2
    parameter2 = val3 / 100.0
    #print(parameter2)


def on_trackbar4(val4):
    global minDistance
    minDistance = val4+1
    print(minDistance)




cv2.namedWindow("test")
cv2.createTrackbar("min radius", "test", 0, 250, on_trackbar1)
cv2.createTrackbar("max radius", "test", 0, 250, on_trackbar2)
cv2.createTrackbar("parameter2", "test", 90, 100, on_trackbar3)
cv2.createTrackbar("minDistance", "test", 10, 99, on_trackbar4)

while (1):

    # cam.get_image(img)
    result, image = cam.read()
    # image = img.get_image_data_numpy()
    # image = cv2.resize(image, (600, 600))

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    high, im = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    low = 0.5 * high

    gray = cv2.medianBlur(gray, 5)

    cany = cv2.Canny(gray, low, high)

    print(parameter2)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT_ALT, 1, minDistance, param1=high, param2=parameter2, minRadius=radiusMin,
                               maxRadius=radiusMax)

    if circles is not None:
        circles = np.uint16(np.around(circles))

        for i in circles[0, :]:
            # draw the outer circle
            cv2.circle(image, (i[0], i[1]), i[2], (0, 255, 0), 2)
            # draw the center of the circle
            cv2.circle(image, (i[0], i[1]), 2, (0, 0, 255), 3)
            iteracia = iteracia + 1

            if iteracia >= 20:
                print("viac ako 20")
                break

    # cv2.imshow("test", np.zeros((100, 100), dtype=np.uint8))
    # slider_value = cv2.getTrackbarPos("Slider", "test")


    cv2.imshow("test", image)
    iteracia = 0
    cv2.imshow("canny", cany)
    a = cv2.waitKey(1)

    if (a == ord('q')):
        break

cam.stop_acquisition()  # stop data acquisition
cam.close_device()  # stop communication
