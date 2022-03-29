# Input Librariesp
from re import sub
from turtle import color
from matplotlib import markers
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os

# set subplot dimensions
rows = 2
cols = 3

# set data plot
fig1, ax = plt.subplots(rows, cols, sharex=True, sharey=True)
for i in range(0, rows):
    for j in range(0, cols):
        # Read from CSV of Model Topology
        caseID = 'Am5_c12'
        # define Topology
        if i == 0:
            model_top = 'Topology '+ str(j+1)
        else:
            model_top = 'Topology '+ str(j+3)
        # read csv
        TopologyFilename = './Topologies/'+caseID+'_'+model_top+'.csv'
        if os.path.exists(TopologyFilename):
            # convert csv to df
            TopologyDataframe = pd.read_csv(TopologyFilename,sep=';')

            # export to txt
            if i == 0:
                mode = 'w'
            else:
                mode = 'a'
            mean = np.mean(TopologyDataframe['Error (%)'])
            std = np.std(TopologyDataframe['Error (%)'])
            max = np.max(TopologyDataframe['Error (%)'])
            f = open('./Topologies/Topologies_Data.txt',mode)
            f.write('Batch: %s\t' % caseID)
            f.write('Topology: %s\t' % model_top)
            f.write('Mean_error: %.3g\t' % mean)
            f.write('Max_error: %.3g\t' % max)
            f.write('Standard Deviation: %.3g\n' % std)
            f.close()

            # data visualization
            ax[i, j].set(title=model_top,
                        xlabel='Theoretical Permeability (-)',
                        ylabel='Estimated Permeability (-)')
            ax[i, j].scatter(TopologyDataframe['Keq/Kpm_teo'],
                            TopologyDataframe['Keq/Kpm_est'], color='royalblue', marker='.')
            ax[i, j].plot(TopologyDataframe['Keq/Kpm_teo'],
                          TopologyDataframe['Keq/Kpm_teo'], color='navy', linewidth=2.0)
            fig1.suptitle(caseID+' Topologies')
            fig1.savefig('./Topologies/'+caseID+'_Topologies.png')
        else:
            break

# set data distribution
fig2, ax2 = plt.subplots(rows, cols, sharex=True, sharey=True)
for i in range(0, rows):
    for j in range(0, cols):
        # define Topology
        if i == 0:
            model_top = 'Topology '+ str(j+1)
        else:
            model_top = 'Topology '+ str(j+3)
        # read csv
        TopologyFilename = './Topologies/'+caseID+'_'+model_top+'.csv'
        if os.path.exists(TopologyFilename):
            # convert csv to df
            TopologyDataframe = pd.read_csv(TopologyFilename,sep=';')

            # data visualization
            ax2[i, j].set(title=model_top,
                        xlabel='Prediction Error (%)',
                        ylabel='Density')
            ax2[i, j] = sns.distplot(np.array(TopologyDataframe['Error (%)']), color = 'm', ax=ax2[i, j])
            fig2.suptitle(caseID+' Topologies Distributions')
            fig2.savefig('./Topologies/'+caseID+'_Distribution.png')
        else:
            break