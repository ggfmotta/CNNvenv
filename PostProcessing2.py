# Input Libraries
from re import sub
from turtle import color
from matplotlib import markers
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from Inputs import *

# read History csv
sensitivity = ['','a','b']
colors = ['b','m','r']
for i in range(0,len(sensitivity)):
        HistoryFilename = './vCNN/History/CNNPerm_'+batchName+threshName+topologyName+sensitivity[i]+'_Hist.csv'
        if os.path.exists(HistoryFilename):
                # convert csv to df
                HistoryDataframe = pd.read_csv(HistoryFilename,sep=',')

                # data visualization
                fig, ax = plt.subplots(nrows = 1, ncols = 2, sharex=False, sharey=False, figsize = (10,6))
                ax[0].set_title('Mean Square Error - Loss Function', fontsize = 10)
                ax[0].plot(HistoryDataframe['loss'], label='Top_'+topologyName+sensitivity[i], color = colors[i])
                ax[0].set_xlabel(xlabel = 'Epochs', fontsize = 10)
                ax[0].legend(loc = "best", fontsize = 8)
                ax[1].set_title('Mean Error (%)', fontsize = 10)
                ax[1].plot(HistoryDataframe['mean_Error'], label='Top_'+topologyName+sensitivity[i], color = colors[i])
                ax[1].set_xlabel(xlabel = 'Epochs', fontsize = 10)
                ax[1].legend(loc = "best", fontsize = 8)
        else:
                break

fig.tight_layout()
fig.savefig('./vCNN/Topologies/Next/Am8_c34_Sensitivity.png')
