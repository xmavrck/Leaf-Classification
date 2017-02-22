import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score,log_loss

import cv2
import os
import glob




###  READ GIVEN CSV DATA BOTH TRAIN AND TEST  ####
LeafTrainData = pd.read_csv("data/train.csv")
LeafTestData = pd.read_csv("data/test.csv")
y = LeafTrainData.ix[:,1]

LeafTrainDataIds = list(LeafTrainData['id'])
LeafTestDataIds = list(LeafTestData['id'])

#### "SPECIES" IS DELETED FROM TRAINING DATA SINCE IT IS THE INDEPENDENT VARIABLE FOR OUR PROBLEM  #####

del LeafTrainData['species']

LeafData = LeafTrainData.append(LeafTestData)
LeafData = LeafData.sort_values('id')

dim = 96

####
columns = ["imgFeat" + str(i) for i in range(1,dim*dim+1)]
imageFeatures = pd.DataFrame(columns=columns)

image_id = []


#### READING IMAGE DATA AND FORMING FEATURES IN THE FORM OF NDARRAY OF DIMENSION 96*96   #######


for filename in glob.glob(os.path.join("data/images", '*.jpg')):
    label = os.path.splitext(os.path.basename(filename))[0]


    image_id = image_id+[label]
    img = cv2.imread(filename,cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (dim,dim), interpolation=cv2.INTER_CUBIC)
    img = img.ravel()
    img = img/max(img)
    imageFeatures.loc[len(imageFeatures)] = img



imageFeatures['id'] = image_id
imageFeatures['id'] = imageFeatures['id'].astype(int)
imageFeatures = imageFeatures.sort_values('id')




##### MERGING THE EXTRACTED IMAGE FEATURES WITH THE GIVEN LEAF FEATURES #####
LeafData = LeafData.merge(imageFeatures,on='id')

LeafTrainDataWithImageFeatures = LeafData.loc[LeafData['id'].isin(LeafTrainDataIds)]
LeafTestDataWithImageFeatures = LeafData.loc[LeafData['id'].isin(LeafTestDataIds)]
X = LeafTrainDataWithImageFeatures.ix[:,1:]
testDataIds = LeafTestData.id




##### THE BELOW HASH-TAGS ARE THE EXPERIMENTS DONE ON VARIOUS MACHINE LEARNING ALGORITHMS FOR THE CROSS-VALIDATION PURPOSE AND RANDOM FOREST IS THE BEST  #######

#X_Train,X_Test,y_Train,y_Test = train_test_split(X,y,test_size= 0.5)

#ModelNN = MLPClassifier()
#ModelRF.fit(X_Train,y_Train)
#ModelLDA = LinearDiscriminantAnalysis()
#ModelGB = GradientBoostingClassifier()


#Prediction = ModelRF.predict(X_Test)
#PredictionProbabilities = ModelRF.predict_proba(X_Test)


#LogLoss = log_loss(y_Test,PredictionProbabilities)
#print(LogLoss)
#print(accuracy_score(Prediction, y_Test))




##### APPLYING RANDOM FOREST MODEL ON COMPLETE TRAINING DATA WITH NUMBER OF TREES BEING 500 AND MAXIMUM FEATURES FOR FOR BEST SPLIT IS 97 SINCE SQUARE ROOT OF NUMBER OF VARIABLES 9409 IS 97 ####

ModelRF = RandomForestClassifier(n_estimators=500,max_features=97,criterion='gini')
ModelRF.fit(X,y)
Prediction = ModelRF.predict(LeafTestDataWithImageFeatures.ix[:,1:])
PredictionProbabilities = ModelRF.predict_proba(LeafTestDataWithImageFeatures.ix[:,1:])

ColumnNames = ModelRF.classes_

FinalDF = pd.DataFrame(PredictionProbabilities,index=testDataIds,columns=ColumnNames)




###### WRITING THE SOLUTION FILE IN REQUIRED FORMAT AS MENTIONED BY KAGGLE #####
FinalDF.to_csv("fourthSubmission.csv")






