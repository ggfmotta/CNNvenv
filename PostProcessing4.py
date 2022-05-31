# Pre-Processing Routines
from matplotlib import markers
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib.ticker as ticker

# Inport csv
modelName = 'Am5_c12_Topology 2'
TestDataframe = pd.read_csv('./vCNN/Topologies/Next/Test/'+modelName+'.csv',sep=';')
TrainDataframe = pd.read_csv('./vCNN/Topologies/Next/Train/'+modelName+'.csv',sep=';')
base = 'AM5C12' # remind to update
top = '_T2_'
pred = 'AM5C34b' # remind to update
TopologyDataframe = pd.read_csv('./vCNN/Topologies/Next/Pred/'+base+top+pred+'.csv',sep=';')

# Data Analysis
fig, ax = plt.subplots(nrows = 2, ncols = 2, sharex=False, sharey=False, figsize = (10,8))
ax[0,0].scatter(TrainDataframe['Keq/Kpm_teo'],
                    TrainDataframe['Keq/Kpm_est'], color='mediumpurple', marker='x', label = 'Am5_c12 Train Data')
ax[0,0].plot(TrainDataframe['Keq/Kpm_teo'],
                        TrainDataframe['Keq/Kpm_teo'], color='indigo', linewidth=2.0)
ax[0,0].scatter(TestDataframe['Keq/Kpm_teo'],
                    TestDataframe['Keq/Kpm_est'], color='lightgreen', marker='x', label = 'Am5_c12 Test Data')
ax[0,0].legend(loc = "upper left", fontsize = 10)                        
ax[0,0].set_ylabel(ylabel = 'Estimated Perm. (-)', fontsize = 10)
ax[0,0].set_xlabel(xlabel = 'Theoretical Perm. (-)', fontsize = 10)

ax[0,1].scatter(TopologyDataframe['Keq/Kpm_teo'],
                    TopologyDataframe['Keq/Kpm_est'], color='lightgreen', marker='x', label = 'Am5_c34 Data')
ax[0,1].plot(TopologyDataframe['Keq/Kpm_teo'],
                        TopologyDataframe['Keq/Kpm_teo'], color='indigo', linewidth=2.0)
ax[0,1].legend(loc = "upper left", fontsize = 10)                        
ax[0,1].set_ylabel(ylabel = 'Estimated Perm. (-)', fontsize = 10)
ax[0,1].set_xlabel(xlabel = 'Theoretical Perm. (-)', fontsize = 10)
#ax[0].set_xlim([1.00, 2.5])
#ax[0].set_ylim([1.00, 2.5])
ax[1,0].scatter(TestDataframe['Keq/Kpm_teo'],TestDataframe['Error (%)'], 
                                        color = 'lightsalmon', label = 'Am5_c12 Test Data')
ax[1,0].axhline(y = TestDataframe['Error (%)'].mean(), color = 'orangered', linestyle = '--', label = 'Mean')
ax[1,0].set_xlabel(xlabel = 'Theoretical Perm. (-)', fontsize = 10)
ax[1,0].set_ylabel(ylabel = 'Error (%)', fontsize = 10)
#ax[1,0].set_xlim([1.00, 1.20])
#ax[1,0].set_ylim([-0.50, 60.00])
ax[1,0].xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
ax[1,0].yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
ax[1,0].legend(loc = "best", fontsize = 8)
ax[1,1].scatter(TopologyDataframe['Keq/Kpm_teo'], TopologyDataframe['Error (%)'], 
                                        color = 'lightsalmon', label = 'Am5_c34 Data')
ax[1,1].axhline(y = TopologyDataframe['Error (%)'].mean(), color = 'orangered', linestyle = '--', label = 'Mean')
ax[1,1].set_xlabel(xlabel = 'Theoretical Perm. (-)', fontsize = 10)
ax[1,1].set_ylabel(ylabel = 'Error (%)', fontsize = 10)
#ax[1,1].set_xlim([1.00, 1.20])
#ax[1,1].set_ylim([-0.50, 60.00])
ax[1,1].xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
ax[1,1].yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
ax[1,1].legend(loc = "best", fontsize = 8)
fig.tight_layout()
fig.savefig('./vCNN/Topologies/'+base+top+pred+'.png')
print('\nEnd\n')