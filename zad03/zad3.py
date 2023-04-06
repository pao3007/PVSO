import cv2
import os

import numpy as np



directory=r'C:\Users\lukac\Desktop\PVSO'
os.chdir(directory)
testImage = cv2.imread('test.jpg')


ys = (int)(testImage.shape[0]/2)
xs = (int)(testImage.shape[1]/2)
testImage = cv2.resize(testImage, (xs, ys))
image = testImage

def on_trackbar1(val1):
    global lineTreshold
    lineTreshold = val1
    #print(lineTreshold)

cv2.namedWindow("Detekcia")
cv2.createTrackbar("LineTreshold", "Detekcia", 900, 1000, on_trackbar1)

def konvolucia(img, filter):
    Ny = img.shape[0]
    Nx = img.shape[1]

    NyZ = Ny - 2
    NxZ = Nx - 2
    image = np.zeros((NyZ, NxZ), dtype = "float64")

    for y in range(1,Ny-1):
        for x in range(1,Nx-1):
            #sum = img[y-1][x-1]*filter[0][0] + img[y-1][x]*filter[0][1] + img[y-1][x+1]*filter[0][2] \
                 # + img[y][x-1]*filter[1][0] + img[y][x]*filter[1][1] + img[y][x+1]*filter[1][2] \
                 # + img[y+1][x-1]*filter[2][0] + img[y+1][x]*filter[2][1] + img[y+1][x+1]*filter[2][2]

            sum = np.sum(img[y - 1:y + 2, x - 1:x + 2] * filter)
            image[y-1][x-1] = sum
    return image

def sopel(img):
    kernelVertical = np.array([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]], dtype = "float32")
    kernelHorizontal = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype = "float32")

    grad_x = konvolucia(img, kernelVertical)
    grad_y = konvolucia(img, kernelHorizontal)

    gradient_magnitude = np.sqrt(np.square(grad_x) + np.square(grad_y))
    gradient_magnitude *= 255.0 / gradient_magnitude.max()

    return gradient_magnitude

def blur3x3(img):
    kernel = np.array([[1/16, 2/16, 1/16],[2/16, 4/16, 2/16],[1/16, 2/16, 1/16]], dtype = "float64")

    filtered = konvolucia(img, kernel)
    return filtered

def my_canny(img, threshold1, threshold2):
    kernelVertical = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype = "float64")
    kernelHorizontal = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype = "float64")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blurred = blur3x3(gray)

    grad_x = konvolucia(blurred, kernelVertical)
    grad_y = konvolucia(blurred, kernelHorizontal)

    gradient_magnitude = np.sqrt(np.square(grad_x) + np.square(grad_y))
    gradient_magnitude *= 255.0 / gradient_magnitude.max()
    cv2.imshow("test blur", grad_x)


    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #blurred = blur3x3(gray)

    grad_x2 = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    grad_y2 = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)

    #gradient_magnitude2 = np.sqrt(np.square(grad_x2) + np.square(grad_y2))
    #gradient_magnitude2 *= 255.0 / gradient_magnitude2.max()
    cv2.imshow("test sopel 2", grad_x2)

    mag, ang = cv2.cartToPolar(grad_x2, grad_y2, angleInDegrees=True)

    non_max = np.zeros_like(mag)
    for i in range(1, mag.shape[0] - 1):
        for j in range(1, mag.shape[1] - 1):
            if (0 <= ang[i, j] < 22.5) or (157.5 <= ang[i, j] <= 180):
                if (mag[i, j] >= mag[i, j - 1]) and (mag[i, j] >= mag[i, j + 1]):
                    non_max[i, j] = mag[i, j]
            elif (22.5 <= ang[i, j] < 67.5):
                if (mag[i, j] >= mag[i - 1, j - 1]) and (mag[i, j] >= mag[i + 1, j + 1]):
                    non_max[i, j] = mag[i, j]
            elif (67.5 <= ang[i, j] < 112.5):
                if (mag[i, j] >= mag[i - 1, j]) and (mag[i, j] >= mag[i + 1, j]):
                    non_max[i, j] = mag[i, j]
            elif (112.5 <= ang[i, j] < 157.5):
                if (mag[i, j] >= mag[i - 1, j + 1]) and (mag[i, j] >= mag[i + 1, j - 1]):
                    non_max[i, j] = mag[i, j]

    thresh_low = threshold1
    thresh_high = threshold2
    strong_edges = (non_max > thresh_high)
    weak_edges = ((non_max >= thresh_low) & (non_max <= thresh_high))

    edges = np.zeros_like(non_max)


    for i in range(1, edges.shape[0] - 1):
        for j in range(1, edges.shape[1] - 1):
            if strong_edges[i, j]:
                edges[i, j] = 255
            elif weak_edges[i, j]:
                if (strong_edges[i - 1:i + 2, j - 1:j + 2].any()):
                    edges[i, j] = 255

    return edges

#test = cv2.cvtColor(testImage, cv2.COLOR_BGR2GRAY)
#test1 = blur3x3(test)
#test2 = sopel(test1)
#cv2.imshow("test blur",test2)

gray = cv2.cvtColor(testImage, cv2.COLOR_BGR2GRAY)
high, im = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
low = 0.5 * high

cany = my_canny(testImage, high, low)

Ny = cany.shape[0]
Nx = cany.shape[1]

Maxdist = int(np.round(np.sqrt(Nx**2 + Ny ** 2)))

thetas = np.deg2rad(np.arange(-90, 90))
rs = np.linspace(-Maxdist, Maxdist, 2*Maxdist)

accumulator = np.zeros((2 * Maxdist, len(thetas)))

for y in range(Ny):
     for x in range(Nx):
          if cany[y,x] > 0:
              for k in range(len(thetas)):
                  r = x * np.cos(thetas[k]) + y * np.sin(thetas[k])
                  accumulator[int(r) + Maxdist, k] += 1

flatAcc = accumulator.flatten()
cv2.imshow("Original", image)

while (1):
    image = cv2.imread('test.jpg')
    ys = (int)(image.shape[0] / 2)
    xs = (int)(image.shape[1] / 2)
    image = cv2.resize(image, (xs, ys))
    a = cv2.waitKey(1)
    if (a == ord('q')):
        break
    for idx in range(flatAcc.shape[0]):
        if flatAcc[idx] > lineTreshold:
            rho = int(rs[int(idx / accumulator.shape[1])])
            theta = thetas[int(idx % accumulator.shape[1])]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = (a * rho)
            y0 = (b * rho)

            x1 = int((x0 + 1000 * (-b)))
            y1 = int((y0 + 1000 * a))
            x2 = int((x0 - 1000 * (-b)))
            y2 = int((y0 - 1000 * a))
            cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), thickness=2)
    cv2.imshow("Detekcia",image)
    print(lineTreshold)
