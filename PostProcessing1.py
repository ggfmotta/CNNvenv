# Input Librariesp
from re import sub
from turtle import color
from matplotlib import markers
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import seaborn as sns
import os

# set subplot dimensions
rows = 2
cols = 3

# set data plot
fig1, ax = plt.subplots(rows, cols, sharex=False, sharey=False)
for i in range(0, rows):
    for j in range(0, cols):
        # Read from CSV of Model Topology
        caseID = 'Am5_c12'
        # define Topology
        if i == 0:
            model_top = 'Topology '+ str(j+1)
        else:
            model_top = 'Topology '+ str(i+j+3)
        # read test csv
        TopologyFilename = './vCNN/Topologies/Train/'+caseID+'_'+model_top+'.csv'
        if os.path.exists(TopologyFilename):
            # convert csv to df
            TopologyDataframe = pd.read_csv(TopologyFilename,sep=';')
            # export to txt
            if i == 0 and j == 0:
                mode = 'w'
            else:
                mode = 'a'
            mean = np.mean(TopologyDataframe['Error (%)'])
            std = np.std(TopologyDataframe['Error (%)'])
            max = np.max(TopologyDataframe['Error (%)'])
            f = open('./vCNN/Topologies/Test/Topologies_Data.txt',mode)
            f.write('Batch: %s\t' % caseID)
            f.write('Topology: %s\t' % model_top)
            f.write('Mean_error: %.3g\t' % mean)
            f.write('Max_error: %.3g\t' % max)
            f.write('Standard Deviation: %.3g\n' % std)
            f.close()
            # data visualization
            ax[i,j].set_title(label = model_top, fontsize = 10)
            ax[i, j].scatter(TopologyDataframe['Keq/Kpm_teo'],
                            TopologyDataframe['Keq/Kpm_est'], color='royalblue', marker='.', label = 'Train Data')
            ax[i, j].plot(TopologyDataframe['Keq/Kpm_teo'],
                          TopologyDataframe['Keq/Kpm_teo'], color='navy', linewidth=2.0)
        else:
            break
        # read test csv
        TestTopologyFilename = './vCNN/Topologies/Test/'+caseID+'_'+model_top+'.csv'
        if os.path.exists(TestTopologyFilename):
            # convert csv to df
            TestTopologyDataframe = pd.read_csv(TestTopologyFilename,sep=';')
            # export to txt
            if i == 0 and j == 0:
                mode = 'w'
            else:
                mode = 'a'
            mean = np.mean(TestTopologyDataframe['Error (%)'])
            std = np.std(TestTopologyDataframe['Error (%)'])
            max = np.max(TestTopologyDataframe['Error (%)'])
            f = open('./vCNN/Topologies/Train/Topologies_Data.txt',mode)
            f.write('Batch: %s\t' % caseID)
            f.write('Topology: %s\t' % model_top)
            f.write('Mean_error: %.4g\t' % mean)
            f.write('Max_error: %.4g\t' % max)
            f.write('Standard Deviation: %.4g\n' % std)
            f.close()
            ax[i, j].scatter(TestTopologyDataframe['Keq/Kpm_teo'],
                            TestTopologyDataframe['Keq/Kpm_est'], color='firebrick', marker='x', label = 'Test Data')
            ax[i,j].legend(loc = "lower right", fontsize = 6.5)
fig1.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.4, hspace=0.3)
fig1.text(0.5, 0.025, 'Theoretical Perm. (-)', ha = 'center', fontsize = 10)
fig1.text(0.025, 0.5, 'Estimated Perm. (-)', va = 'center', rotation = 'vertical', fontsize = 10)
fig1.suptitle(caseID+' Topologies')
fig1.savefig('./vCNN/Topologies/'+caseID+'_Topologies.png')
        
# set data distribution
fig2, ax2 = plt.subplots(rows, cols, sharex=False, sharey=False)
for i in range(0, rows):
    for j in range(0, cols):
        # define Topology
        if i == 0:
            model_top = 'Topology '+ str(j+1)
        else:
            model_top = 'Topology '+ str(i+j+3)
        # read csv
        TopologyFilename = './vCNN/Topologies/Test/'+caseID+'_'+model_top+'.csv'
        if os.path.exists(TopologyFilename):
            # convert csv to df
            TopologyDataframe = pd.read_csv(TopologyFilename,sep=';')
            # data visualization
            ax2[i,j].set_title(label = model_top, fontsize = 10)
            ax2[i, j] = sns.histplot(np.array(TopologyDataframe['Keq/Kpm_est'])-np.array(TopologyDataframe['Keq/Kpm_teo']),
                                        color = 'm',
                                        stat = "density", common_norm=False, kde = True,
                                        ax = ax2[i, j],
                                        )
            ax2[i,j].set_ylabel(ylabel = " ")
            ax2[i,j].xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))        
        else:
            break
fig2.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.4, hspace=0.3)
fig2.text(0.5, 0.025, 'Error (-)', ha = 'center', fontsize = 10)
fig2.text(0.025, 0.5, 'Density', va = 'center', rotation = 'vertical', fontsize = 10)
fig2.suptitle(caseID+' Topologies Test Data Normal Distribution')
fig2.savefig('./vCNN/Topologies/'+caseID+'_Distribution.png')