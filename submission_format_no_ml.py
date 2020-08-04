import os
import pandas as pd

# read video names from folder
directory = "./test"
files = os.listdir(directory)
videos = [file for file in files if os.path.splitext(file)[1] == '.mp4']

# predictions
pred = pd.read_csv("test_predictions_method1-nomedian.csv", sep=',')
pred_list = pred['predictions'].to_list()
pred_df = pd.DataFrame(zip(videos,pred_list), columns = ["filename","stalled"])

pred_df.to_csv("submission2.csv", encoding='utf-8', index=False)
