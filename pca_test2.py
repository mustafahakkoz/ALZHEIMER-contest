from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import imutils
import time
import dlib
import cv2
from math import atan2, cos, sin, sqrt, pi, degrees


def blend_images(frame_list):
   for idx, img in enumerate(frame_list,1):
      if idx == 1:
         first_img = img
         continue
      else:
        second_img = img
        second_weight = 1/(idx+1)
        first_weight = 1 - second_weight
        first_img = cv2.addWeighted(first_img, first_weight, second_img, second_weight, 0)
   return first_img


def drawAxis(img, p_, q_, colour, scale):
    p = list(p_)
    q = list(q_)
    
    angle = atan2(p[1] - q[1], p[0] - q[0]) # angle in radians
    hypotenuse = sqrt((p[1] - q[1]) * (p[1] - q[1]) + (p[0] - q[0]) * (p[0] - q[0]))
    # Here we lengthen the arrow by a factor of scale
    q[0] = p[0] - scale * hypotenuse * cos(angle)
    q[1] = p[1] - scale * hypotenuse * sin(angle)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv2.LINE_AA)
    # create the arrow hooks
    p[0] = q[0] + 9 * cos(angle + pi / 4)
    p[1] = q[1] + 9 * sin(angle + pi / 4)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv2.LINE_AA)
    p[0] = q[0] + 9 * cos(angle - pi / 4)
    p[1] = q[1] + 9 * sin(angle - pi / 4)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv2.LINE_AA)
    
def getOrientation(pts, img):
    
    sz = len(pts)
    data_pts = np.empty((sz, 2), dtype=np.float64)
    for i in range(data_pts.shape[0]):
        data_pts[i,0] = pts[i,0,0]
        data_pts[i,1] = pts[i,0,1]
    # Perform PCA analysis
    mean = np.empty((0))
    mean, eigenvectors, eigenvalues = cv2.PCACompute2(data_pts, mean)
    # Store the center of the object
    cntr = (int(mean[0,0]), int(mean[0,1]))
    
    
    cv2.circle(img, cntr, 3, (255, 0, 255), 2)
    p1 = (cntr[0] + 0.02 * eigenvectors[0,0] * eigenvalues[0,0], cntr[1] + 0.02 * eigenvectors[0,1] * eigenvalues[0,0])
    p2 = (cntr[0] - 0.02 * eigenvectors[1,0] * eigenvalues[1,0], cntr[1] - 0.02 * eigenvectors[1,1] * eigenvalues[1,0])
    drawAxis(img, cntr, p1, (0, 255, 0), 1)
    drawAxis(img, cntr, p2, (255, 255, 0), 5)
    angle = atan2(eigenvectors[0,1], eigenvectors[0,0]) # orientation in radians
    
    return angle

# start the video stream thread
print("[INFO] starting video stream thread...")
cap = cv2.VideoCapture('./images and samples/100109.mp4')


time.sleep(1.0)
(grabbed, frame) = cap.read()
#cv2.imshow("Image", frame)
#cv2.waitKey(0)

gaussian = cv2.GaussianBlur(frame, (3, 3), 0)
frame_HSV = cv2.cvtColor(gaussian, cv2.COLOR_BGR2HSV)
orange_border = cv2.inRange(frame_HSV, (0, 100, 100), (50, 255, 255))
#cv2.imshow("Image2", orange_border)
#cv2.waitKey(0)

black_border = cv2.threshold(orange_border, 100, 150, cv2.THRESH_BINARY)[1]
#cv2.imshow("black_border", black_border)
#cv2.waitKey(0)


# find contours (i.e., outlines) of the foreground objects in the
# thresholded image
cnts, _ = cv2.findContours(black_border.copy(), cv2.RETR_LIST,
	cv2.CHAIN_APPROX_NONE)
#sort contıurs
min_idx=0
min_area=cv2.contourArea(cnts[0])
for i, c in enumerate(cnts):
    # Calculate the area of each contour
    area = cv2.contourArea(c)
    if area < min_area:
        min_area = area
        min_idx = i
c=cnts[min_idx]
src = black_border.copy()
cv2.drawContours(src, [c], -1, (255, 255, 255), -1)
cv2.imshow("Contours", src)
cv2.waitKey(0)

mask = cv2.threshold(src, 200, 255, cv2.THRESH_BINARY)[1]
cv2.imshow("mask", mask)
cv2.waitKey(0)
#--------------------------------PCA---------
contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
c = contours[0]
sz = len(c)
data_pts = np.empty((sz, 2), dtype=np.float64)
for i in range(data_pts.shape[0]):
    data_pts[i,0] = c[i,0,0]
    data_pts[i,1] = c[i,0,1]
# Perform PCA analysis
mean = np.empty((0))
mean, eigenvectors, eigenvalues = cv2.PCACompute2(data_pts, mean)
# Store the center of the object
cntr = (int(mean[0,0]), int(mean[0,1]))

cv2.circle(mask, cntr, 3, (255, 0, 255), 2)
p1 = (cntr[0] + 0.02 * eigenvectors[0,0] * eigenvalues[0,0], cntr[1] + 0.02 * eigenvectors[0,1] * eigenvalues[0,0])
p2 = (cntr[0] - 0.02 * eigenvectors[1,0] * eigenvalues[1,0], cntr[1] - 0.02 * eigenvectors[1,1] * eigenvalues[1,0])
drawAxis(mask, cntr, p1, (0, 255, 0), 1)
drawAxis(mask, cntr, p2, (255, 255, 0), 5)
angle_mask0 = atan2(eigenvectors[0,1], eigenvectors[0,0]) # orientation in radians
angle_mask1 = atan2(eigenvectors[1,1], eigenvectors[1,0]) # orientation in radians
print("eigenvector0 of mask (angle): ", angle_mask0*180/np.pi if angle_mask0>0 else angle_mask0*180/np.pi+180)
print("eigenvector1 of mask (angle): ", angle_mask1*180/np.pi if angle_mask1>0 else angle_mask1*180/np.pi+180)

cv2.imshow('pca', mask)
cv2.waitKey(0)
#--------------------------------------------

gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
output = cv2.bitwise_and(gray, gray, mask=mask)
cv2.imshow('output', output)
cv2.waitKey(0)

summation = output.copy()

frame_list= []

while True:
	# if this is a file video stream, then we need to check if
	# there any more frames left in the buffer to process
   (grabbed, frame) = cap.read()
   if not grabbed:
      break
	# grab the frame from the threaded video file stream, resize
	# it, and convert it to grayscale
	# channels)
   #cv2.imshow("frame", frame)
   #cv2.waitKey(0)
#   gaussian = cv2.GaussianBlur(frame, (3, 3), 0)
   gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
   output = cv2.bitwise_and(gray, gray, mask=mask)
   #cv2.imshow("output", output)
   #cv2.waitKey(0)
   
   #summation = cv2.addWeighted(summation,0.95,output,0.05,0)
   #cv2.imshow("summation", summation)
   #cv2.waitKey(0)
   frame_list.append(output)
#   windowlist=frame_list[-10:]
#   blended_windowlist = blend_images(windowlist)


# deneme kısmı ----------
windowlist=frame_list[0:5]
blended_windowlist = blend_images(windowlist)
cv2.imshow("windowlist", blended_windowlist)
cv2.waitKey(0)
#  ----------


blended = blend_images(frame_list)
cv2.imshow("blended", blended)
cv2.waitKey(0)

sharpened = cv2.threshold(blended_windowlist, 65, 255, cv2.THRESH_BINARY)[1]
cv2.imshow("sharpened", sharpened)
cv2.waitKey(0)

# otsu
blur = cv2.GaussianBlur(sharpened,(5,5),0)
th3,ret3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
cv2.imshow("ret3", ret3)
cv2.waitKey(0)
print(th3)

# closing
# kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
kernel=np.ones((3,3),np.uint8)
closing = cv2.morphologyEx(ret3, cv2.MORPH_CLOSE, kernel)
cv2.imshow("closing", closing)
cv2.waitKey(0)

# opening
# kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
kernel=np.ones((2,2),np.uint8)
opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
cv2.imshow("opening", opening)
cv2.waitKey(0)

#--------------------------------PCA---------
contours, _ = cv2.findContours(opening, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
area_list=[]


for i, c in enumerate(contours):
    # Calculate the area of each contour
    area = cv2.contourArea(c)
    area_list.append(area)
    
max_idx = np.argmax(area_list)
c = contours[max_idx]
sz = len(c)
data_pts = np.empty((sz, 2), dtype=np.float64)
for i in range(data_pts.shape[0]):
    data_pts[i,0] = c[i,0,0]
    data_pts[i,1] = c[i,0,1]
# Perform PCA analysis
mean = np.empty((0))
mean, eigenvectors, eigenvalues = cv2.PCACompute2(data_pts, mean)
# Store the center of the object
cntr = (int(mean[0,0]), int(mean[0,1]))

cv2.circle(opening, cntr, 3, (0, 0, 0), 2)
p1 = (cntr[0] + 0.02 * eigenvectors[0,0] * eigenvalues[0,0], cntr[1] + 0.02 * eigenvectors[0,1] * eigenvalues[0,0])
p2 = (cntr[0] - 0.02 * eigenvectors[1,0] * eigenvalues[1,0], cntr[1] - 0.02 * eigenvectors[1,1] * eigenvalues[1,0])
drawAxis(opening, cntr, p1, (255, 255, 255), 15)
drawAxis(opening, cntr, p2, (255, 255, 255), 30)
angle0 = atan2(eigenvectors[0,1], eigenvectors[0,0]) # orientation in radians
angle1 = atan2(eigenvectors[1,1], eigenvectors[1,0]) # orientation in radians

print("eigenvector0 of biggest component (angle): ", angle0*180/np.pi if angle0>0 else angle0*180/np.pi+180)
print("eigenvector1 of biggest component (angle): ", angle1*180/np.pi if angle1>0 else angle1*180/np.pi+180)
cv2.imshow('pca2', opening)
cv2.waitKey(0)
#--------------------------------------------