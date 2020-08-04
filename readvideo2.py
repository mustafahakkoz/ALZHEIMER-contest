import pandas as pd
import numpy as np
import imutils
import time
import cv2


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

    
def getOrientation(pts):
    sz = len(pts)
    data_pts = np.empty((sz, 2), dtype=np.float64)
    for i in range(data_pts.shape[0]):
        data_pts[i,0] = pts[i,0,0]
        data_pts[i,1] = pts[i,0,1]
    # Perform PCA analysis
    mean = np.empty((0))
    mean, eigenvectors, eigenvalues = cv2.PCACompute2(data_pts, mean)
    angle = cv2.phase(eigenvectors[1,1], eigenvectors[1,0],angleInDegrees=True)
    # convert angle to 0-180
    angle180 = angle[0][0] if angle[0][0]<180 else angle[0][0]-180
    return angle180


def orientationDiff(angle1, angle2):
    diff = abs(angle1 - angle2)
    diff90 = diff if diff<90 else 180-diff
    return diff90

    
# start the video stream thread
print("[INFO] starting video stream thread...")
cap = cv2.VideoCapture('./images and samples/100109.mp4')

time.sleep(1.0)
(grabbed, frame) = cap.read()
#cv2.imshow("Image", frame)
#cv2.waitKey(0)

# find orange border
gaussian = cv2.GaussianBlur(frame, (3, 3), 0)
frame_HSV = cv2.cvtColor(gaussian, cv2.COLOR_BGR2HSV)
orange_border = cv2.inRange(frame_HSV, (0, 100, 100), (50, 255, 255))
cv2.imshow("Image2", orange_border)
cv2.waitKey(0)

# make border a bit bolder
kernel = np.ones((3,3),np.uint8)
dilation = cv2.dilate(orange_border,kernel,iterations = 1)
#cv2.imshow("dilation", dilation)
#cv2.waitKey(0)

# make border grey
grey_border = cv2.threshold(dilation, 100, 150, cv2.THRESH_BINARY)[1]
cv2.imshow("grey_border", grey_border)
cv2.waitKey(0)

# find contours (i.e., outlines) of the foreground objects in the
# thresholded image
cnts = cv2.findContours(grey_border.copy(), cv2.RETR_LIST,
	cv2.CHAIN_APPROX_NONE)
cnts = imutils.grab_contours(cnts)
# find true contour (inside of mask)
if len(cnts) <= 1:
    src =  np.zeros((384, 512, 1), np.uint8)
else:
    cnts_df = pd.DataFrame([cv2.contourArea(c) for c in cnts], columns = ['area'])
    idx = cnts_df.sort_values(by=['area'], ascending=False).iloc[[1]].index[0]
    c=cnts[idx]
    src = grey_border.copy()
    cv2.drawContours(src, [c], -1, (255, 255, 255), -1)
#    cv2.imshow("Contours", src)
#    cv2.waitKey(0)

mask = cv2.threshold(src, 200, 255, cv2.THRESH_BINARY)[1]
cv2.imshow("mask", mask)
cv2.waitKey(0)

#---------------PCA-mask--------------------
contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
c = contours[0]
# Calculate the angle of mask
angle_mask = getOrientation(c)
#--------------------------------------------

#gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#output = cv2.bitwise_and(gray, gray, mask=mask)
#summation = output.copy()

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
#   equ = cv2.equalizeHist(output)
#   clahe = cv2.createCLAHE(clipLimit=65, tileGridSize=(8,8))
#   c1 = clahe.apply(output)
   #cv2.imshow("output", output)
   #cv2.waitKey(0)
   
   #summation = cv2.addWeighted(summation,0.95,output,0.05,0)
   #cv2.imshow("summation", summation)
   #cv2.waitKey(0)
   frame_list.append(output)
#   windowlist=frame_list[-10:]
#   blended_windowlist = blend_images(windowlist)


# some testing ----------
windowlist=frame_list[50:60]
blended_windowlist = blend_images(windowlist)
#cv2.imshow("windowlist", blended_windowlist)
#cv2.waitKey(0)
#  ----------------------

#################### SIMPLE BLENDING ####################
blended = blend_images(frame_list)
cv2.imshow("blended", blended)
cv2.waitKey(0)

#equ = cv2.equalizeHist(blended)
#clahe = cv2.createCLAHE(clipLimit=65, tileGridSize=(8,8))
#c1 = clahe.apply(output)
#cv2.imshow("equ", c1)
#cv2.waitKey(0)

sharpened = cv2.threshold(blended, 65, 255, cv2.THRESH_BINARY)[1]
cv2.imshow("sharpened", sharpened)
cv2.waitKey(0)

#
#th3 = cv2.adaptiveThreshold(blended,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
#cv2.THRESH_BINARY,11,-2)
#cv2.imshow("th3", th3)
#cv2.waitKey(0)

# otsu
#blur = cv2.GaussianBlur(blended,(5,5),0)
#th_otsu, otsu = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#cv2.imshow("otsu", otsu)
#cv2.waitKey(0)

#median = cv2.medianBlur(th3,5)
#cv2.imshow("median", median)
#cv2.waitKey(0)

#gaussian = cv2.GaussianBlur(th3, (3, 3), 0)
#cv2.imshow("gaussian", gaussian)
#cv2.waitKey(0)

# [laplacian]
# Apply Laplace function
#dst = cv2.Laplacian(sharpened, 0, ksize=3)
#abs_dst = cv2.convertScaleAbs(dst)
#cv2.imshow("abs_dst", abs_dst)
#cv2.waitKey(0)

# otsu
blur = cv2.GaussianBlur(sharpened,(5,5),0)
th_otsu, otsu = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
cv2.imshow("otsu", otsu)
cv2.waitKey(0)


# closing
# kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
kernel=np.ones((3,3),np.uint8)
closing = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, kernel)
cv2.imshow("closing", closing)
cv2.waitKey(0)

# opening
# kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
kernel=np.ones((2,2),np.uint8)
opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
cv2.imshow("opening", opening)
cv2.waitKey(0)


##erosion
#kernel = np.ones((1,1),np.uint8)
#erosion = cv2.erode(ret3,kernel,iterations = 1)
#cv2.imshow("erosion", erosion)
#cv2.waitKey(0)
#
##dilation
#kernel = np.ones((3,3),np.uint8)
#dilation = cv2.dilate(erosion,kernel,iterations = 1)
#cv2.imshow("dilation", dilation)
#cv2.waitKey(0)

#remove little contours
eliminated_contours = []
contours, _ = cv2.findContours(opening, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
for c in contours:
    # Calculate the area of each contour
    area = cv2.contourArea(c)
    # Filter contours that are too small or too large
    if area < 1e2 or 1e5 < area:
        eliminated_contours.append(c)
#        continue
#    angle = getOrientation(c)
    # Filter contours which have different orientations than mask
#    if (orientationDiff(angle,angle_mask) > 45):
#        eliminated_contours.append(c)
#------------------------------------------
# fill black unfiltered contours 
cv2.drawContours(opening, eliminated_contours, -1, (0, 0, 0), -1)
cv2.imshow("clean_opening", opening)
cv2.waitKey(0)

    
#################### CONTOUR ORIENTATITON METHOD 1: blending contours ####################
filtered_flows=[]
window = 10
for i in range(len(frame_list)-window-1):
    flow = blend_images(frame_list[i : i+window])
#    equ = cv2.equalizeHist(flow)
#    clahe = cv2.createCLAHE(clipLimit=65, tileGridSize=(8,8))
#    c1 = clahe.apply(flow)
    # threshold = 65
    sharpened = cv2.threshold(flow, 65, 255, cv2.THRESH_BINARY)[1]
    # otsu
    blur = cv2.GaussianBlur(sharpened,(5,5),0)
    th,otsu = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#    # closing
#    kernel=np.ones((3,3),np.uint8)
#    closing = cv2.morphologyEx(ret3, cv2.MORPH_CLOSE, kernel)
#    # opening
#    kernel=np.ones((2,2),np.uint8)
#    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
#    cv2.imshow("flow", opening)
#    cv2.waitKey(0)
    result = otsu
    #--------------PCA of contours -----------
    eliminated_contours = []
    contours, _ = cv2.findContours(result, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    if len(contours)==0:
        continue
    for c in contours:
        # Calculate the area of each contour
        area = cv2.contourArea(c)
        # Filter contours that are too small or too large
        if area < 1e2 or 1e5 < area:
            eliminated_contours.append(c)
            continue
        angle = getOrientation(c)
        # Filter contours which have different orientation than mask
        if (orientationDiff(angle,angle_mask) > 45):
            eliminated_contours.append(c)
    #------------------------------------------
    # fill black unfiltered contours 
    cv2.drawContours(result, eliminated_contours, -1, (0, 0, 0), -1)
    # add result to list 
    filtered_flows.append(result)

# blend filtered frames
blended_flows = blend_images(filtered_flows)
cv2.imshow("blended_flows", blended_flows)
cv2.waitKey(0)


# threshold = 25
gaus = cv2.GaussianBlur(blended_flows,(5,5),0)
basicth = cv2.threshold(gaus, 25, 255, cv2.THRESH_BINARY)[1]
cv2.imshow("basicth", basicth)
cv2.waitKey(0)

# adaptive
adaptive = cv2.adaptiveThreshold(blended_flows,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,-2)
cv2.imshow("adaptive", adaptive)
cv2.waitKey(0)

# otsu
gaus = cv2.GaussianBlur(blended_flows,(5,5),0)
th_otsu2,otsu2 = cv2.threshold(gaus,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
cv2.imshow("otsu2", otsu2)
cv2.waitKey(0)

# otsu
gaus = cv2.GaussianBlur(blended_flows,(5,5),0)
basicth2 = cv2.threshold(gaus, 65, 255, cv2.THRESH_BINARY)[1]
th_otsu25,otsu25 = cv2.threshold(basicth2,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
cv2.imshow("otsu25", otsu25)
cv2.waitKey(0)

#remove little contours
eliminated_contours = []
contours, _ = cv2.findContours(otsu2, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
for c in contours:
    # Calculate the area of each contour
    area = cv2.contourArea(c)
    # Filter contours that are too small or too large
    if area < 1e2 or 1e5 < area:
        eliminated_contours.append(c)
#        continue
#    angle = getOrientation(c)
    # Filter contours which have different orientations than mask
#    if (orientationDiff(angle,angle_mask) > 45):
#        eliminated_contours.append(c)
#------------------------------------------
# fill black unfiltered contours 
cv2.drawContours(otsu2, eliminated_contours, -1, (0, 0, 0), -1)
cv2.imshow("clean_otsu2", otsu2)
cv2.waitKey(0)

#################### CONTOUR ORIENTATITON METHOD 2: blending flows ####################
filtered_frames=[]
window = 5
for i in range(len(frame_list)-window-1):
    flow = blend_images(frame_list[i : i+window])
#    equ = cv2.equalizeHist(flow)
#    clahe = cv2.createCLAHE(clipLimit=65, tileGridSize=(8,8))
#    c1 = clahe.apply(flow)
    # threshold = 65
    sharpened = cv2.threshold(flow, 65, 255, cv2.THRESH_BINARY)[1]
    # otsu
    blur = cv2.GaussianBlur(sharpened,(5,5),0)
    th,otsu = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#    # closing
#    kernel=np.ones((3,3),np.uint8)
#    closing = cv2.morphologyEx(ret3, cv2.MORPH_CLOSE, kernel)
#    # opening
#    kernel=np.ones((2,2),np.uint8)
#    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
#    cv2.imshow("flow", opening)
#    cv2.waitKey(0)
    result = otsu
    #--------------PCA of biggest contour -----------
    contours, _ = cv2.findContours(result, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    if len(contours)==0:
        continue
    area_list=[]
    for c in contours:
        # Calculate the area of each contour
        area = cv2.contourArea(c)
        area_list.append(area)
    # Check if biggest contour has different orientation than mask
    max_idx = np.argmax(area_list)
    angle = getOrientation(contours[max_idx])
    
#    print("max idx:", max_idx, "angle:", angle, "difference to mask:", orientationDiff(angle,angle_mask))
#    cv2.imshow("result", result)
#    cv2.waitKey(0)
    
    if (orientationDiff(angle,angle_mask) < 45):
#        middle_frame =  int((window+i)/2)+1
#        filtered_frames.append(frame_list[middle_frame])
        filtered_frames.append(flow)
    #------------------------------------------

# blend filtered frames
blended_flows2 = blend_images(filtered_frames)
cv2.imshow("blended_flows2", blended_flows2)
cv2.waitKey(0)

# threshold = 65
basicth2 = cv2.threshold(blended_flows2, 65, 255, cv2.THRESH_BINARY)[1]
cv2.imshow("basicth2", basicth2)
cv2.waitKey(0)

# adaptive
adaptive2 = cv2.adaptiveThreshold(blended_flows2,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,-2)
cv2.imshow("adaptive2", adaptive2)
cv2.waitKey(0)

# otsu
gaus2 = cv2.GaussianBlur(blended_flows2,(5,5),0)
th_otsu3,otsu3 = cv2.threshold(gaus2,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
cv2.imshow("otsu3", otsu3)
cv2.waitKey(0)

# otsu
gaus2 = cv2.GaussianBlur(basicth2,(5,5),0)
th_otsu35,otsu35 = cv2.threshold(gaus2,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
cv2.imshow("otsu35", otsu35)
cv2.waitKey(0)

#remove little contours
eliminated_contours = []
contours, _ = cv2.findContours(otsu35, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
for c in contours:
    # Calculate the area of each contour
    area = cv2.contourArea(c)
    # Filter contours that are too small or too large
    if area < 1e2 or 1e5 < area:
        eliminated_contours.append(c)
#        continue
#    angle = getOrientation(c)
    # Filter contours which have different orientations than mask
#    if (orientationDiff(angle,angle_mask) > 45):
#        eliminated_contours.append(c)
#------------------------------------------
# fill black unfiltered contours 
cv2.drawContours(otsu35, eliminated_contours, -1, (0, 0, 0), -1)
cv2.imshow("clean_otsu35", otsu35)
cv2.waitKey(0)