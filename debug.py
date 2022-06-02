# Input Libraries
from re import sub
from turtle import color
from matplotlib import markers
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

fig2, ax2 = plt.subplots(nrows = 1, ncols = 1, sharex=False, sharey=False, figsize = (8,6))
# Read from CSV of Model Topology
caseID = 'Am8_c34'
model_top = 'Topology 2c' # 2B
# read test csv
TopologyFilename = './vCNN/Topologies/Next/Train/'+caseID+'_'+model_top+'.csv'
if os.path.exists(TopologyFilename):
        # convert csv to df
        TopologyDataframe = pd.read_csv(TopologyFilename,sep=';')
        # export to txt
        mode = 'a'
        mean = np.mean(TopologyDataframe['Error (%)'])
        std = np.std(TopologyDataframe['Error (%)'])
        max = np.max(TopologyDataframe['Error (%)'])
        f = open('./vCNN/Topologies/Debug_Data.txt',mode)
        f.write('Batch: %s\t' % caseID)
        f.write('Type: Train\t')
        f.write('Topology: %s\t' % model_top)
        f.write('Mean_error: %.3g\t' % mean)
        f.write('Max_error: %.3g\t' % max)
        f.write('Standard Deviation: %.3g\n' % std)
        f.close()
        # data visualization
        #ax2.set_title(label = caseID + ' ' + model_top + 'B Performance', fontsize = 12)
        ax2.scatter(TopologyDataframe['Keq/Kpm_teo'],
                        TopologyDataframe['Keq/Kpm_est'], color='mediumpurple', marker='.', label = 'Train Data')
        ax2.plot(TopologyDataframe['Keq/Kpm_teo'],
                        TopologyDataframe['Keq/Kpm_teo'], color='indigo', linewidth=2.0)
        ax2.set_ylabel(ylabel = 'Estimated Perm. (-)', fontsize = 10)
        ax2.set_xlabel(xlabel = 'Theoretical Perm. (-)', fontsize = 10)
else:
        pass

# read test csv
TestTopologyFilename = './vCNN/Topologies/Next/Test/'+caseID+'_'+model_top+'.csv'
if os.path.exists(TestTopologyFilename):
# convert csv to df
        TestTopologyDataframe = pd.read_csv(TestTopologyFilename,sep=';')
        mean = np.mean(TestTopologyDataframe['Error (%)'])
        std = np.std(TestTopologyDataframe['Error (%)'])
        max = np.max(TestTopologyDataframe['Error (%)'])
        mode = 'a'
        f = open('./vCNN/Topologies/Debug_Data.txt',mode)
        f.write('Batch: %s\t' % caseID)
        f.write('Type: Test\t')
        f.write('Topology: %s\t' % model_top)
        f.write('Mean_error: %.4g\t' % mean)
        f.write('Max_error: %.4g\t' % max)
        f.write('Standard Deviation: %.4g\n' % std)
        f.close()
        ax2.scatter(TestTopologyDataframe['Keq/Kpm_teo'],
                TestTopologyDataframe['Keq/Kpm_est'], 
                color='lightgreen', marker='x', label = 'Test Data')
        ax2.legend(loc = "upper left", fontsize = 10)
            
fig2.tight_layout()
fig2.savefig('./vCNN/Topologies/'+caseID+'_'+model_top+'.png')
