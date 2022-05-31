# Pre-Processing Routines
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
subset = am8_c12

# Prepare Data df
imFiles, Porous, Perms, PMPerms = read_perm_data("/home/gmotta/CNN/Data/AMs_data.csv",delimiter=",", imlist = subset)
Dataframe = pd.read_csv("/home/gmotta/CNN/Data/AMs_data.csv",sep=',')
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
test_size = 0.5
XDEL, Xinfo, YDEL, Ytrue = train_test_split(X,y,test_size=test_size)
Xinfo = X
Ytrue = y

# Load Model
print('\nLoading Model\n')
modelName = 'CNNPerm_5_34_2a'
save_path = '/home/gmotta/CNN/vCNN/SavedModels/'+modelName+'/'
model_latest = keras.models.load_model(save_path+'my_model.h5', custom_objects={'mean_Error': mean_Error})
model_latest.compile(optimizer='adam',loss='mse',metrics=mean_Error)
print('\nLoaded Model\n')

# Predict
print('\nPredicting...\n')
Ypred = model_latest.predict(x = Xinfo)
print('\nPredicted Test Data\n')

# Check Test Results
TopologyDataframe = pd.DataFrame(columns=['Keq/Kpm_teo','Keq/Kpm_est','Error (%)'])
TopologyDataframe['Keq/Kpm_teo'] = Ytrue
TopologyDataframe['Keq/Kpm_est'] = Ypred
TopologyDataframe['Error (%)'] = 100*abs(TopologyDataframe['Keq/Kpm_est']-TopologyDataframe['Keq/Kpm_teo'])/TopologyDataframe['Keq/Kpm_teo']
print('\nChecked Results\n')

# Export to csv
base = 'AM5C34' # remind to update
top = '_T2_'
pred = 'AM8C12' # remind to update
TopologyDataframe.to_csv('./vCNN/Topologies/Next/Pred/'+base+top+pred+'.csv',sep=';')

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