# Leaf Classification for 99 varieties of leaves

# DATA:
-The data is provided by kaggle. (ref: https://www.kaggle.com/c/leaf-classification/data)

-train.csv and test.csv contains features of 99 varieties of leaves
-Images folder contains the respective images with the id's matching the  image file names. 

# Problem Statement:
-The taining data is to be trained by classifier to classify the test data and give the probabilities for each data point of being 
in a particular class.

# PROCESS:
-The given data has various features in a csv file of that of leaves along with the images of the leaves.
-Features for the image files are extracted to later combine with the other given features.
-The result will write a csv file which is in the format mentioned by kaggle for the submission purpose.

# Requirements:
 - Python3.4
 - pandas
 - scipy
 - sklearn
 - cv2
 - glob
 
### Output
 - Result will be saved as File  ``` fourthSubmission.csv ```