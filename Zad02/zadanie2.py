import cv2
import numpy
import glob
from ximea import xiapi
import os

cam = xiapi.Camera() #use xiapi cam
cam.open_device()   #open camera

#camera settings
cam.set_exposure(10000)
cam.set_param('imgdataformat','XI_RGB32')
cam.set_param('auto_wb', 1)


directory=r'/home/d618/Desktop/cv2'  #set up directory for saving photos
os.chdir(directory)     #get into working directory
img = xiapi.Image()     #create instance to store image
cam.start_acquisition() #start data inquisiton
a = ""

iteracia = 1

while(1):

    cam.get_image(img)

    image = img.get_image_data_numpy()
    image = cv2.resize(image,(600,600))
    cv2.imshow("test", image)
    a = cv2.waitKey(1)


    if(a == ord('c')):

        filename = "img" + str(iteracia) + ".jpg"
        cv2.imwrite(filename, image)

        iteracia = iteracia + 1

    if (a == ord('q')):
        break

cam.stop_acquisition()  # stop data acquisition
cam.close_device()  # stop communication

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = numpy.zeros((7*5,3), numpy.float32)
objp[:,:2] = numpy.mgrid[0:5,0:7].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
images = glob.glob('/home/d618/Desktop/cv2/*.jpg')
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (5,7), None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)
        # Draw and display the corners

        cv2.drawChessboardCorners(img, (5,7), corners2, ret)
        cv2.imshow('img', img)
        cv2.waitKey(2000)

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

print('fx = ',mtx[0,0])
print('fy = ',mtx[1,1])
print('cx = ',mtx[0,2])
print('cy = ',mtx[1,2])