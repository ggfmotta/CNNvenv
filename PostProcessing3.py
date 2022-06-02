# Pre-Processing Routines
from telnetlib import X3PAD
from tensorflow import keras
from PreProcessing import *
from matplotlib import markers
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.model_selection import train_test_split

# Read Images and CSV into CNN Inputs; Features and Labels
all_files = os.listdir("/home/gmotta/CNN/Images/")
all_files_noExt = [s[0:len(s)-4] for s in all_files]

am5 = [s for s in all_files if "am5_" in s]
am5_x = [s for s in am5 if "_x" in s]
am5_y = [s for s in am5 if "_y" in s]
am5_c1 = [s for s in am5 if "c1" in s]
am5_c1x = [s for s in am5_c1 if "_x" in s]
am5_c1y = [s for s in am5_c1 if "_y" in s]
am5_c12 = [s for s in am5 if "c1" in s or "c2" in s]
am5_c12x = [s for s in am5_c12 if "_x" in s]
am5_c12y = [s for s in am5_c12 if "_y" in s]
am5_c123 = [s for s in am5 if "c1" in s or "c2" in s or "c3" in s]
am5_c123x = [s for s in am5_c123 if "_x" in s]
am5_c123y = [s for s in am5_c123 if "_y" in s]
am5_c34 = [s for s in am5 if "c3" in s or "c4" in s]
am5_c34x = [s for s in am5_c34 if "_x" in s]
am5_c34y = [s for s in am5_c34 if "_y" in s]
am5_c1x = [s for s in am5_c1 if "_x" in s]
am5_c12x = [s for s in am5_c12 if "_x" in s]
am5_c12y = [s for s in am5_c12 if "_y" in s]
am8 = [s for s in all_files if "am8_" in s]
am8_c12 = [s for s in am8 if "c1" in s or "c2" in s]
am8_c34 = [s for s in am8 if "c3" in s or "c4" in s]
am8_c12x = [s for s in am8_c12 if "_x" in s]
am8_c12y = [s for s in am8_c12 if "_y" in s]

# Define subset
subset = am5_c34

# Prepare Data df
imFiles, Porous, Perms, PMPerms = read_perm_data("/home/gmotta/CNN/Data/AMs_data2.csv",delimiter=",", imlist = subset)
Dataframe = pd.read_csv("/home/gmotta/CNN/Data/AMs_data2.csv",sep=',')
Dataframe.dropna()

# List of Columns to Keep
ColumnsToKeep = ['Image_file','Sample','Slice','%Area','Perim.','Keq/Kpm']

# Drop entire Columns not in this list
ListOfColumnsToDrop = [s for s in Dataframe.columns if s not in ColumnsToKeep]
Dataframe.drop(ListOfColumnsToDrop,1,inplace=True)

# Filter Subset of Images
Dataframe['Image_file'] = Dataframe['Image_file'] + '.tif' #added
Dataframe = Dataframe[Dataframe['Image_file'].isin(subset)]
Dataframe = Dataframe.reindex(np.random.permutation(Dataframe.index))

# Define Test Data
X,y,Info = create_NN_data("/home/gmotta/CNN/Images/",\
                        imFiles = Dataframe['Image_file'].values,\
                        Target = Dataframe['Keq/Kpm'].values,\
                        Extra = Dataframe['%Area'].values/100,\
                        imgSize = imgSize)

# Split subset data
#test_size = 0.5
#Xtrain, Xtest, Ytrain, Ytest  = train_test_split(X,y,test_size=test_size)
Xinfo = X#Xtrain
Ytrue = y#Ytrain

# Load Model
print('\nLoading Model\n')
modelName = 'CNNPerm_8_34_2c'
subsetID = 'Am8_c34'
modelTop = 'Topology 2c'
save_path = '/home/gmotta/CNN/vCNN/SavedModels/'+modelName+'/'
model_latest = keras.models.load_model(save_path+'my_model.h5', custom_objects={'mean_Error': mean_Error})
model_latest.compile(optimizer='adam',loss='mse',metrics=mean_Error)
print('\nLoaded Model\n')
'''
# Predict
print('\nPredicting Train Data...\n')
Ypred = model_latest.predict(x = Xtrain)
print('\nPredicted Train Data\n')

# Check Train Results
TrainDataframe = pd.DataFrame(columns=['Keq/Kpm_teo','Keq/Kpm_est','Error (%)'])
TrainDataframe['Keq/Kpm_teo'] = Ytrain
TrainDataframe['Keq/Kpm_est'] = Ypred
TrainDataframe['Error (%)'] = 100*abs(TrainDataframe['Keq/Kpm_est']-TrainDataframe['Keq/Kpm_teo'])/TrainDataframe['Keq/Kpm_teo']
TrainDataframe.to_csv('./vCNN/Topologies/Next/Train/'+subsetID+'_'+modelTop+'.csv',sep=';')
print('\nChecked Results\n')
'''
# Predict
print('\nPredicting Test Data...\n')
Ypred2 = model_latest.predict(x = Xinfo)
print('\nPredicted Train Data\n')

# Check Train Results
TestDataframe = pd.DataFrame(columns=['Keq/Kpm_teo','Keq/Kpm_est','Error (%)'])
TestDataframe['Keq/Kpm_teo'] = Ytrue
TestDataframe['Keq/Kpm_est'] = Ypred2
TestDataframe['Error (%)'] = 100*abs(TestDataframe['Keq/Kpm_est']-TestDataframe['Keq/Kpm_teo'])/TestDataframe['Keq/Kpm_teo']
# Export to csv
base = 'AM8C34' # remind to update
top = '_T2_'
pred = 'AM5C34' # remind to update
TestDataframe.to_csv('/home/gmotta/CNN/vCNN/Topologies/Next/Pred/'+base+top+pred+'.csv',sep=';')
print('\nChecked Results\n')

'''
# Export to txt
path2txt = './vCNN/Topologies/Next/Pred/Topologies_Data.txt'
mode = 'w'
if os.path.exists(path2txt): mode = 'a'
mean = np.mean(TopologyDataframe['Error (%)'])
std = np.std(TopologyDataframe['Error (%)'])
max = np.max(TopologyDataframe['Error (%)'])
f = open(path2txt,mode)
f.write('Base subset: %s\t' % base)
f.write('Topology: %s\t' % top)
f.write('Pred subset: %s\t' %pred)
f.write('Mean_error: %.3g\t' % mean)
f.write('Max_error: %.3g\t' % max)
f.write('Standard Deviation: %.3g\n' % std)
f.close()
'''