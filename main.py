from __future__ import division, print_function, absolute_import

# Pre-Processing Routines
from PreProcessing import *

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

# change subset and change Model Configurations in Inputs.py
subset = am8_c34
caseID = 'Am8_c34'

imFiles, Porous, Perms, PMPerms = read_perm_data("/home/gmotta/CNN/Data/AMs_data.csv",delimiter=",", imlist = subset)
# Using Pandas Dataframes
Dataframe = pd.read_csv("/home/gmotta/CNN/Data/AMs_data.csv",sep=',')

# Drop entire rows with NA values
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
                        Dataframe.Image_file.values,\
                        Dataframe['Keq/Kpm'].values,\
                        Extra=Dataframe['%Area'].values/100,\
                        imgSize=imgSize)

# Model Topology Definition
model = ModelTopology(X)

# Train Model Topology
# Regular Train
model_top = 'Topology '+ topologyName
whole_time = time.time()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)
X_test = np.array(X_test)
y_test = np.array(y_test)
X_train = np.array(X_train)
y_train = np.array(y_train)

print('\nTraining model\n')
history = model.fit(x = X_train,y = y_train,batch_size = 10,epochs = 1000,validation_split = 0.15,callbacks = [checkpointer,es])

# Save model
#print('\nSaving model\n')
#save_path = '/home/gmotta/CNN/vCNN/SavedModels/'+modelName+'/'
#model.save(save_path+'model.h5')
#print('\nSaved model\n')

# predict train
print('\nPredicting...\n')
y_trainPred = model.predict(x = X_train)
print('\nPredicted Train Data\n')
# predict test
y_testPred = model.predict(x = X_test)
print('\nPredicted Test Data\n')
end_time = time.time() - whole_time

# Check Train Results
TrainDataframe = pd.DataFrame(columns=['Keq/Kpm_teo','Keq/Kpm_est','Error (%)'])
TrainDataframe['Keq/Kpm_teo'] = y_train
TrainDataframe['Keq/Kpm_est'] = y_trainPred
TrainDataframe['Error (%)'] = 100*abs(TrainDataframe['Keq/Kpm_est']-TrainDataframe['Keq/Kpm_teo'])/TrainDataframe['Keq/Kpm_teo']
# Export to csv
TrainDataframe.to_csv('./vCNN/Topologies/Next/Train/'+caseID+'_'+model_top+'.csv',sep=';')

# Check Test Results
TopologyDataframe = pd.DataFrame(columns=['Keq/Kpm_teo','Keq/Kpm_est','Error (%)'])
TopologyDataframe['Keq/Kpm_teo'] = y_test
TopologyDataframe['Keq/Kpm_est'] = y_testPred
TopologyDataframe['Error (%)'] = 100*abs(TopologyDataframe['Keq/Kpm_est']-TopologyDataframe['Keq/Kpm_teo'])/TopologyDataframe['Keq/Kpm_teo']
# Export to csv
TopologyDataframe.to_csv('./vCNN/Topologies/Next/Test/'+caseID+'_'+model_top+'.csv',sep=';')

# Convert the history.history dict to a pandas DataFrame 
hist_df = pd.DataFrame(history.history)
# Save to csv
with open('/home/gmotta/CNN/vCNN/History/'+modelName+'_Hist.csv', mode='w') as f:
    hist_df.to_csv(f, index=False)

print(' ')
print('Case Topology: %s' % model_top)
print('Case Batch: %s' % caseID)
print("Total time = %g [s]\n" % end_time)