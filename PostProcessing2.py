# Input Libraries
from re import sub
from turtle import color
from matplotlib import markers
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Read from CSV of results
originalFilename = './Data/AMs_data.csv'
# resulfsFilename = './Results/Train_Am5_c123_Area0-20_Test_Same_TestGroupEval.csv'

originalDataframe = pd.read_csv(originalFilename,sep=',')
all_files = os.listdir("Images")
all_files_noExt = [s[0:len(s)-4] for s in all_files]
am5 = [s for s in all_files_noExt if "am5_" in s]
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
am8 = [s for s in all_files if "am5_" in s]
am8_c12 = [s for s in am8 if "c1" in s or "c2" in s]
am8_c34 = [s for s in am8 if "c3" in s or "c4" in s]
am8_c12x = [s for s in am8_c12 if "_x" in s]
am8_c12y = [s for s in am8_c12 if "_y" in s]

case = am5
subset = am5_c1
caseID = 'Am5'
subsetID = 'Am5_c1'
# caseId = 'Train_'+caseID+'_Test_Same_TestGroupEval'
caseId = 'Train_ValSplit_15_'+caseID+'_Test_Same_TestGroupEval'

# Fill Dataframe    
trainedDataframe = pd.read_csv('./Results/'+caseId+'.csv', sep=';')
#resultImages = resultsDataframe.Image_filename.values
resultImages = list(originalDataframe[originalDataframe.Image_file.isin(case)]['Image_file'])

# Read Training Data
subsetDataframe = originalDataframe[originalDataframe.Image_file.isin(subset)]
#trainingArea = subsetDataframe['%Area'][subsetDataframe.Image_file.isin(resultImages)]
#trainingArea = list(subsetDataframe[subsetDataframe.Image_file.isin(subset)]['%Area'])
testPerm = subsetDataframe['Keq/Kpm'][subsetDataframe.Image_file.isin(resultImages)]
#trainingPerm = list(subsetDataframe[subsetDataframe.Image_file.isin(subset)]['Keq/Kpm'])
trainedPerm = trainedDataframe['Estimated Perm Inc']
meanAccuracy = 100*abs(testPerm - trainedPerm)/testPerm

plt.figure()
plt.scatter(trainedPerm, testPerm, color = 'indigo', marker = 'o')
plt.plot(trainedPerm, trainedPerm, color = 'violet', linewidth = 2.0)
plt.xlabel('Original Permeability (-)', fontsize=14)
plt.ylabel('Estimated Permeability (-)', fontsize=14)
plt.text(1.05, 2.05, 'Mean error: ' + '{0:.2f}'.format(meanAccuracy) + '%', fontsize = 10)
plt.title(caseID+' w/ '+subsetID+' subset', fontsize=20)
plt.savefig('./Results/Train_'+caseID+'_Test_'+subsetID+' subset.png')


'''
fig2, ax = plt.subplots(1, 2)#, gridspec_kw={"width_ratios":[1,1, 0.05]})
h0 = ax[0].hist2d(trainingArea, trainingPerm, bins=(20, 20), cmap=plt.cm.jet)
ax[0].set(title = 'Training Set',
        xlabel = 'Keq/Kpm',
        ylabel = 'Porousity %')
plt.colorbar(h0[3],ax=ax[0],orientation='horizontal')
h1 = ax[1].hist2d(testArea, testPerm, bins=(20, 20), cmap=plt.cm.jet)
ax[1].set(title = 'Min Test Accuracy: {:.2f} %'.format(minAccuracy),
        xlabel = 'Keq/Kpm')
plt.colorbar(h1[3],ax=ax[1],orientation='horizontal')
# fig.colorbar(h0[3], cax=ax[2], ax=ax)
fig2.savefig('./Results/'+caseId + '_Histograms.png')
'''