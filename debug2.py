# Pre-Processing Routines
from matplotlib import markers
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib.ticker as ticker

# Inport csv
modelName = 'Am8_c34_Topology 2c'
TestDataframe = pd.read_csv('./vCNN/Topologies/Next/Test/'+modelName+'.csv',sep=';')
TrainDataframe = pd.read_csv('./vCNN/Topologies/Next/Train/'+modelName+'.csv',sep=';')

# Data Analysis
fig, ax = plt.subplots(nrows = 1, ncols = 2, sharex=False, sharey=False, figsize = (10,5))
ax[0].scatter(TrainDataframe['Keq/Kpm_teo'],
                    TrainDataframe['Keq/Kpm_est'], color='mediumpurple', marker='x', label = 'Am8_c34 Train Data')
ax[0].plot(TrainDataframe['Keq/Kpm_teo'],
                        TrainDataframe['Keq/Kpm_teo'], color='indigo', linewidth=2.0)
ax[0].scatter(TestDataframe['Keq/Kpm_teo'],
                    TestDataframe['Keq/Kpm_est'], color='lightgreen', marker='x', label = 'Am8_c34 Test Data')
ax[0].legend(loc = "upper left", fontsize = 10)                        
ax[0].set_ylabel(ylabel = 'Estimated Perm. (-)', fontsize = 10)
ax[0].set_xlabel(xlabel = 'Theoretical Perm. (-)', fontsize = 10)
#ax[0,0].set_xlim([1.00, 2.5])
#ax[0].set_ylim([1.00, 2.5])
ax[1].scatter(TestDataframe['Keq/Kpm_teo'],TestDataframe['Error (%)'], 
                                        color = 'lightsalmon', label = 'Am8_c34 Test Data')
ax[1].axhline(y = TestDataframe['Error (%)'].mean(), color = 'orangered', linestyle = '--', label = 'Mean')
ax[1].set_xlabel(xlabel = 'Theoretical Perm. (-)', fontsize = 10)
ax[1].set_ylabel(ylabel = 'Error (%)', fontsize = 10)
#ax[1].set_xlim([1.00, 1.20])
#ax[1].set_ylim([-0.50, 100.00])
ax[1].xaxis.set_major_formatter(ticker.FormatStrFormatter('%.0f'))
ax[1].yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
ax[1].legend(loc = "best", fontsize = 8)

fig.tight_layout()
fig.savefig('./vCNN/Topologies/'+modelName+'_Error.png')
print('\nEnd\n')