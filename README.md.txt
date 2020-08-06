## Clog Loss: Advance Alzheimer’s Research with Stall Catchers

A submission for  [drivendata.com 3D-MRI video contest.](https://www.drivendata.org/competitions/65/clog-loss-alzheimers-research/page/207/)

team: ru_kola  
members: [@mustafahakkoz](https://github.com/mustafahakkoz), [@Aysenuryilmazz](https://github.com/Aysenuryilmazz)
rank: 44 / 922  
score (matthew's correlation coefficient): 0.1564  
dataset: [Stall Catchers micro dataset](https://www.drivendata.org/competitions/65/clog-loss-alzheimers-research/page/217/#videos) 2399 3D-MRI videos with 70/30 class ratio (flowing/stalled)  
<img src="https://github.com/mustafahakkoz/ALZHEIMER-contest/blob/master/images%20and%20samples/ezgif-6-a26e6587b744.gif">

The main idea of the project is preprocessing videos hardly by blending all frames in to single one with different approaches, then applying a CNN classifier on them.

### PROCESSING VIDEOS AND SIMPLE BLENDING METHOD  
<img src="https://github.com/mustafahakkoz/ALZHEIMER-contest/blob/master/images%20and%20samples/Resim1.png">

### CONTOUR ORIENTATION METHOD 1: Blending Contours  
<img src="https://github.com/mustafahakkoz/ALZHEIMER-contest/blob/master/images%20and%20samples/Resim2.png">

### CONTOUR ORIENTATION METHOD 2: Blending Frames  
<img src="https://github.com/mustafahakkoz/ALZHEIMER-contest/blob/master/images%20and%20samples/Resim3.png">

### CNN CLASSIFIER  
3 convolution+maxpooling dual layers followed by 1 fully-connected layer just as described in [keras.io blog](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html) trained on [kaggle's gpu kernels.](https://www.kaggle.com/hakkoz/alz-cnn-method1-nomedian)  

### FOLDER CONTENT  
- **Processing datasets and to images with 3 methods defined above:** This code also produces predictions without ML, just by counting contours. [process_videos.py](https://github.com/mustafahakkoz/ALZHEIMER-contest/blob/master/process_videos.py)  
- **CNN predictor:** [CNN.py](https://github.com/mustafahakkoz/ALZHEIMER-contest/blob/master/CNN.py)  
- **Submission format for no-ML predictions:** [submission_format_no_ml.py](https://github.com/mustafahakkoz/ALZHEIMER-contest/blob/master/submission_format_no_ml.py)  
- **Submission format for CNN predictions:** [submission_format_cnn_predictor.py](https://github.com/mustafahakkoz/ALZHEIMER-contest/blob/master/submission_format_cnn_predictor.py)  
- **Weights of trained CNN model:** [method1-nomedian-wholedataset.h5](https://github.com/mustafahakkoz/ALZHEIMER-contest/blob/master/method1-nomedian-wholedataset.h5)  
- **PCA debugging code for contour orientation method 1:** [pca_test.py](https://github.com/mustafahakkoz/ALZHEIMER-contest/blob/master/pca_test.py)  
- **PCA debugging code for contour orientation method 2:** [pca_test2.py](https://github.com/mustafahakkoz/ALZHEIMER-contest/blob/master/pca_test2.py)  
- **Debugging code for whole preprocessing pipeline:** [readvideo2.py](https://github.com/mustafahakkoz/ALZHEIMER-contest/blob/master/readvideo2.py)  
- **A sample video and output images:** [./images and samples/](https://github.com/mustafahakkoz/ALZHEIMER-contest/tree/master/images%20and%20samples)  



