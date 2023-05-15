from ximea import xiapi
import cv2
import os
import numpy

cam = xiapi.Camera() #use xiapi cam
cam.open_device()   #open camera

#camera settings
cam.set_exposure(10000)
cam.set_param('imgdataformat','XI_RGB32')
cam.set_param('auto_wb', 1)


directory=r'/home/d618/Desktop/cv1'  #set up directory for saving photos

os.chdir(directory)     #get into working directory

img = xiapi.Image()     #create instance to store image

cam.start_acquisition() #start data inquisiton
a = ""

iteracia = 1

while(1):

    cam.get_image(img)

#   if cv2.waitKey() != ord(' '): #cv2.waitKey(1):
#
#   cam.get_image(img)
    image = img.get_image_data_numpy()
    image = cv2.resize(image,(600,600))
    cv2.imshow("test", image)
    a = cv2.waitKey(1)


    if(a == ord('c')):

        filename = "img" + str(iteracia) + ".jpg"
        cv2.imwrite(filename,image)

        iteracia = iteracia + 1

    if(iteracia == 5):

        #uloha 4
        image = cv2.imread('img2.jpg')
        height, width = image.shape[:2]
        rotated_image = numpy.zeros((width, height, 3), dtype=numpy.uint8)

        for x in range(width):
            for y in range(height):
                rotated_image[x, y] = image[height - y - 1, x]

        #cv2.imshow('rotated_image', rotated_image)

        #uloha 5
        image = cv2.imread('img3.jpg')
        red = image.copy()
        red[:, :, 0] = 0
        red[:, :, 1] = 0
        #cv2.imshow('red-rgb', red)

        #uloha2
        img1 = cv2.imread('img1.jpg')
        img4 = cv2.imread('img4.jpg')
        im_v1 = cv2.vconcat([img1, rotated_image])
        im_v2 = cv2.vconcat([img4, red])
        im_h = cv2.hconcat([im_v1, im_v2])
        #cv2.imshow('mozaika', im_h)


        # uloha 3
        kernel = numpy.array([[-1, 0, 1],
                              [-2, 0, 2],
                              [-1, 0, 1]], numpy.float32)
        im_h[0:600, 0:600] = cv2.filter2D(im_h[0:600, 0:600], -1, kernel)
        #cv2.imwrite(kernel_image, im_h)

        #uloha 6
        height, width = im_h.shape[:2]
        dimensions = im_h.shape
        print('Image Dimension: ', dimensions)
        print('Image Height: ', height)
        print('Image Width: ', width)
        print('Image Size: ', im_h.size)
        print('Image Type: ', im_h.dtype)


        filename = 'mozaika.jpg'
        cv2.imshow('mozaika', im_h)
        cv2.imwrite(filename, im_h)

        iteracia = iteracia + 1



    if(a == ord('q')):
        break


cam.stop_acquisition()  # stop data acquisition
cam.close_device()  # stop communication
