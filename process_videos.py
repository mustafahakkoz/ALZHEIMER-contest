import numpy as np
import pandas as pd
import imutils
import time
import cv2
import os
from sklearn.metrics import matthews_corrcoef
#################### UTILITY FUNCTIONS ####################
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


def write_image(directory, output_folder, image_name, image):
    if not os.path.exists(os.path.join(directory,output_folder)):
        os.mkdir(os.path.join(directory,output_folder)) 
    cv2.imwrite(os.path.join(directory, output_folder, clean_name+'.jpg'), image )
    
    
def read_video(video_path):
    # start the video stream thread
    cap = cv2.VideoCapture(video_path)
    
    time.sleep(1.0)
    (grabbed, frame) = cap.read()
    
    # find orange border
    gaussian = cv2.GaussianBlur(frame, (3, 3), 0)
    frame_HSV = cv2.cvtColor(gaussian, cv2.COLOR_BGR2HSV)
    orange_border = cv2.inRange(frame_HSV, (0, 100, 100), (50, 255, 255))
    
    # make border a bit bolder
    kernel = np.ones((3,3),np.uint8)
    dilation = cv2.dilate(orange_border,kernel,iterations = 1)
    
    # make border grey
    grey_border = cv2.threshold(dilation, 100, 150, cv2.THRESH_BINARY)[1]
    
    # find contours (i.e., outlines) of the foreground objects in the
    # thresholded image
    cnts = cv2.findContours(grey_border.copy(), cv2.RETR_LIST,
    	cv2.CHAIN_APPROX_NONE)
    cnts = imutils.grab_contours(cnts)
    src = grey_border.copy()
    # find true contour (inside of mask)
    if len(cnts) <= 1:
        return  np.zeros((384, 512, 1), np.uint8), 0, 0
    cnts_df = pd.DataFrame([cv2.contourArea(c) for c in cnts], columns = ['area'])
    idx = cnts_df.sort_values(by=['area'], ascending=False).iloc[[1]].index[0]
    c=cnts[idx]
    
    cv2.drawContours(src, [c], -1, (255, 255, 255), -1)
    
    mask = cv2.threshold(src, 200, 255, cv2.THRESH_BINARY)[1]

    #---------------PCA-mask--------------------
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    c = contours[0]
    # Calculate the angle of mask
    angle_mask = getOrientation(c)
    # Calculate area of mask
    area_mask = cv2.contourArea(c)
    #--------------------------------------------

    # read frames and add to a list
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
       gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
       output = cv2.bitwise_and(gray, gray, mask=mask)
       frame_list.append(output)
    return frame_list, angle_mask, area_mask


#################### SIMPLE BLENDING METHOD ####################
# just blend all frames
def simple_blending_method(frame_list):
    blended = blend_images(frame_list)
    
    # simple threshold = 65
    sharpened = cv2.threshold(blended, 65, 255, cv2.THRESH_BINARY)[1]
    
    # otsu
    blur = cv2.GaussianBlur(sharpened,(5,5),0)
    th_otsu, otsu = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    # closing
    kernel=np.ones((3,3),np.uint8)
    closing = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, kernel)
    
    # opening
    kernel=np.ones((2,2),np.uint8)
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)

    #remove little contours
    eliminated_contours = []
    contours, _ = cv2.findContours(opening, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    for c in contours:
        # Calculate the area of each contour
        area = cv2.contourArea(c)
        # Filter contours that are too small or too large
        if area < 1e2 or 1e5 < area:
            eliminated_contours.append(c)
            
    # fill black unfiltered contours 
    cv2.drawContours(opening, eliminated_contours, -1, (0, 0, 0), -1)
    return opening


#################### CONTOUR ORIENTATITON METHOD 1: blending contours ####################
# construct short flows by blending frames with a window, check contour orientations in flows, 
# if they match with orientation of mask, blend contours
def contour_orientation_method1(frame_list, angle_mask, area_mask):
    filtered_flows=[]
    window = 5
    for i in range(len(frame_list)-window-1):
        flow = blend_images(frame_list[i : i+window])
        # median blur
        median = cv2.medianBlur(flow,7)
        # threshold = 65
        sharpened = cv2.threshold(median, 65, 255, cv2.THRESH_BINARY)[1]
        # otsu
        blur = cv2.GaussianBlur(sharpened,(5,5),0)
        th,otsu = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        #--------------PCA of contours -----------
        eliminated_contours = []
        contours, _ = cv2.findContours(otsu, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        if len(contours)==0:
            continue
        for c in contours:
            # Calculate the area of each contour
            area = cv2.contourArea(c)
            # Filter contours that are too small or too large
            if area < area_mask*0.01 or area_mask <= area:
                eliminated_contours.append(c)
                continue
            angle = getOrientation(c)
            # Filter contours which have different orientation than mask
            if (orientationDiff(angle,angle_mask) > 45):
                eliminated_contours.append(c)
        #------------------------------------------
        # fill black eliminated contours 
        cv2.drawContours(otsu, eliminated_contours, -1, (0, 0, 0), -1)
        # add result to list 
        filtered_flows.append(otsu)
    # comment here to see errors
    if len(filtered_flows) == 0:
        return np.zeros((384, 512, 1), np.uint8)
    # blend filtered frames
    blended_flows = blend_images(filtered_flows)
    
    # otsu
    gaus = cv2.GaussianBlur(blended_flows,(5,5),0)
    th_otsu,otsu = cv2.threshold(gaus,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    #remove little contours
    eliminated_contours = []
    contours, _ = cv2.findContours(otsu, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    for c in contours:
        # Calculate the area of each contour
        area = cv2.contourArea(c)
        # Filter contours that are too small or too large
        if area < area_mask*0.01 or area_mask <= area:
            eliminated_contours.append(c)

    # fill black  eliminated contours 
    cv2.drawContours(otsu, eliminated_contours, -1, (0, 0, 0), -1)
    return otsu


#################### CONTOUR ORIENTATITON METHOD 2: blending frames ####################
# construct short flows by blending frames with a window, check orientation of biggest contour in flows, 
# if it match with orientation of mask, blend flows
def contour_orientation_method2(frame_list, angle_mask, area_mask):
    filtered_frames=[]
    window = 10
    for i in range(len(frame_list)-window-1):
        flow = blend_images(frame_list[i : i+window])
#        clahe = cv2.createCLAHE(clipLimit=65, tileGridSize=(8,8))
#        c1 = clahe.apply(flow)
        # threshold = 65
        sharpened = cv2.threshold(flow, 65, 255, cv2.THRESH_BINARY)[1]
        # otsu
        blur = cv2.GaussianBlur(sharpened,(5,5),0)
        th,otsu = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        #--------------PCA of biggest contour -----------
        contours, _ = cv2.findContours(otsu, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
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
 
        if (orientationDiff(angle,angle_mask) < 45):
            filtered_frames.append(flow)
        #------------------------------------------
    # comment here to see errors
    if len(filtered_frames) == 0:
        return np.zeros((384, 512, 1), np.uint8)
    # blend filtered frames
    blended_flows = blend_images(filtered_frames)
    
    # threshold = 65
    basicth = cv2.threshold(blended_flows, 65, 255, cv2.THRESH_BINARY)[1]
    
    # otsu
    gaus = cv2.GaussianBlur(basicth,(5,5),0)
    th_otsu,otsu = cv2.threshold(gaus,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    #remove little contours
    eliminated_contours = []
    contours, _ = cv2.findContours(otsu, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    for c in contours:
        # Calculate the area of each contour
        area = cv2.contourArea(c)
        # Filter contours that are too small or too large
        if area < area_mask*0.01 or area_mask <= area :
            eliminated_contours.append(c)

    # fill black unfiltered contours 
    cv2.drawContours(otsu, eliminated_contours, -1, (0, 0, 0), -1)
    return otsu


#################### MAIN ####################
directory = "./micro"
output_folder = "method1-nomedian"

files = os.listdir(directory)
videos = [file for file in files if os.path.splitext(file)[1] == '.mp4']

# read data
metadata_df = pd.read_csv("./metadata/train_metadata.csv", sep=',')
labels_df = pd.read_csv("./metadata/train_labels.csv", sep=',')
#tier1_micro = [video for video in videos if video in metadata_df.loc[metadata_df['tier1']==True, 'filename'].to_list()]
#tier1_micro_stalled = [video for video in tier1_micro if video in labels_df.loc[labels_df['stalled']==1, 'filename'].to_list()]

predictions = []
contour_counts=[]
for i, video_name in enumerate(videos):
    print("[INFO] processing index: ", i, " video name: ", video_name, " is started.")
    clean_name = os.path.splitext(video_name)[0]
    file_path = os.path.join(directory,video_name)
    frame_list, angle_mask, area_mask = read_video(file_path)
    image = contour_orientation_method1(frame_list, angle_mask, area_mask)
    write_image(directory, output_folder, clean_name, image)
    print("       ... done.")
    # predictions
    contours, _ = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    contour_counts.append(len(contours))
    if len(contours)>1:
        predictions.append(1)
    else:
        predictions.append(0)

# evaluation
true_labels_micro = [labels_df.loc[labels_df['filename']==video, 'stalled'].squeeze() for video in videos]
pred_score_micro = matthews_corrcoef(true_labels_micro, predictions)
print(pred_score_micro)

# export outputs
predictions_df = pd.DataFrame(zip(contour_counts,predictions), columns = ["contour_counts","predictions"])
predictions_df.to_csv("predictions_method1-nomedian.csv", encoding='utf-8')

# 0.23694613809322196 ->contour_orientation_method1 no median
# 0.22344012669581176 -> contour_orientation_method1 median
# 0.2662261064410542 ->contour_orientation_method2
# 0.17265817672643158 ->simple
        
# 0.09691587345338028 -> cnn-basic contour_orientation_method1 no median
# 0.10080577717127491 -> cnn complex - dropout(0.5)
# 0.1158313820809831 -> cnn complex - no dropout 0.12617454075567316
# 0.1648599173123011 -> cnn complex - no dense 
# 0.06664070419513618 -> epoch 30
        
# 0.05488729658746616 -> cnn method2 no dropout epoch 3
        
# 0.1272300121627428 -> cnn method1 median dropout epoch 3
# 0.02491895841036883 -> cnn method1 median dropout epoch 3 no dense