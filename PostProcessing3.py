# Pre-Processing Routines
from tensorflow import keras
from PreProcessing import *
from matplotlib import markers
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
#from Inputs import modelName, batchName, threshName, topologyName

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

# define subset, X, y
subset = am5_c12
imFiles, Porous, Perms, PMPerms = read_perm_data("Data/AMs_data.csv",delimiter=",", imlist = subset)
Dataframe = pd.read_csv("/home/gmotta/CNN/Data/AMs_data.csv",sep=',')
Dataframe.dropna()
# Filter Subset of Images
Dataframe['Image_file'] = Dataframe['Image_file'] + '.tif' #added

list_size = len(subset)
test_size = 0.85
size = round(test_size*list_size)
Xinfo = []
Ytrue = []

for i in range(0,size):
    file = random.choice(subset)
    Xinfo.append(file)

Ytrue = Dataframe[Dataframe['Image_file'].isin(Xinfo)]['Keq/Kpm']

# Define Test Data
X,y,Info = create_NN_data("/home/gmotta/CNN/Images/",\
                        imFiles = Dataframe[Dataframe['Image_file'].isin(Xinfo)]['Image_file'].values,\
                        Target = Dataframe[Dataframe['Image_file'].isin(Xinfo)]['Keq/Kpm'].values,\
                        Extra=Dataframe['%Area'].values/100,\
                        imgSize=imgSize)

# Load Model
modelName = 'CNNPerm_5_12_2'
save_path = '/home/gmotta/CNN/vCNN/SavedModels/'+modelName+'/'
model_latest = keras.models.load_model(save_path+'model.h5', compile = False)
print('\nLoaded Model\n')

Ypred = model_latest.predict(x = X)
print('\nPredicted Test Data\n')

# Check Test Results
TopologyDataframe = pd.DataFrame(columns=['Keq/Kpm_teo','Keq/Kpm_est','Error (%)'])
TopologyDataframe['Keq/Kpm_teo'] = Ytrue
TopologyDataframe['Keq/Kpm_est'] = Ypred
TopologyDataframe['Error (%)'] = 100*abs(TopologyDataframe['Keq/Kpm_est']-TopologyDataframe['Keq/Kpm_teo'])/TopologyDataframe['Keq/Kpm_teo']
print('\nChecked Results\n')

fig, ax = plt.subplots(nrows = 1, ncols = 1, sharex=False, sharey=False, figsize = (8,6))
ax.set_title('Topology 2 Prediction Test', fontsize = 12)
ax.scatter(TopologyDataframe['Keq/Kpm_teo'],
                    TopologyDataframe['Keq/Kpm_est'], color='lightgreen', marker='.', label = 'Test Data')
ax.plot(TopologyDataframe['Keq/Kpm_teo'],
                        TopologyDataframe['Keq/Kpm_teo'], color='indigo', linewidth=2.0)
ax.legend(loc = "upper left", fontsize = 10)                        
ax.set_ylabel(ylabel = 'Estimated Perm. (-)', fontsize = 10)
ax.set_xlabel(xlabel = 'Theoretical Perm. (-)', fontsize = 10)
ax.set_ylabel(ylabel = 'Estimated Perm. (-)', fontsize = 10)
ax.set_xlabel(xlabel = 'Theoretical Perm. (-)', fontsize = 10)
fig.tight_layout()
fig.savefig('./vCNN/Topologies/Next/Pred/Am5_c12_Topology2_Am5_c34.png')

print('\nEnd\n')