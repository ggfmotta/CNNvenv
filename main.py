from __future__ import division, print_function, absolute_import

# Pre-Processing Routines
from PreProcessing import *

# (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
# input_shape = (28, 28, 1)

# X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
# X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

# X_train = X_train.astype('float32')
# X_test = X_test.astype('float32')

# X_train /= 255
# X_test /= 255

# X = X_train # X = []
# y = y_train # y = []


# Read Images and CSV into CNN Inputs; Features and Labels
all_files = os.listdir("Images")
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
subset = am5_c12
caseID = 'Am5_c12'
#caseId = 'Train_ValSplit_15_'+caseID+'_Test_Same_TestGroupEval'

imFiles, Porous, Perms, PMPerms = read_perm_data("Data/AMs_data.csv",delimiter=",", imlist = am5_c12)
# Using Pandas Dataframes
Dataframe = pd.read_csv("Data/AMs_data.csv",sep=',')

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

# mask = np.random.rand(len(Dataframe)) < 0.85

# # Separate Training and Testing Sets from Dataframe
# TrainingDataframe = pd.DataFrame(Dataframe[mask])
# ResultDataframe = pd.DataFrame(Dataframe[~mask])

# Define Test Data
X,y,Info = create_NN_data('Images',\
                        Dataframe.Image_file.values,\
                        Dataframe['Keq/Kpm'].values,\
                        Extra=Dataframe['%Area'].values/100,\
                        imgSize=imgSize)
# X_train,y_train,Info_train = create_NN_data('Images',\
#                                             TrainingDataframe.Image_file.values,\
#                                             TrainingDataframe['Keq/Kpm'].values,\
#                                             Extra=TrainingDataframe['%Area'].values/100,\
#                                             imgSize=imgSize)

# X_test,y_test,Info_test = create_NN_data('Images',\
#                                             ResultDataframe.Image_file.values,\
#                                             ResultDataframe['Keq/Kpm'].values,\
#                                             Extra=ResultDataframe['%Area'].values/100,\
#                                             imgSize=imgSize)

# Save CNN Inputs: Features X
# pickle_out = open('CNNInputs_X.pickle','wb')
# pickle.dump(X,pickle_out)
# pickle_out.close()

# Save CNN Inputs: Labels y
# pickle_out = open('CNNInputs_y.pickle','')
# pickle.dump(y,pickle_out)
# pickle_out.clo()

# # Load CNN Inputs: Features X
# pickle_in = open('CNNInputs_X.pickle','rb')
# X = pickle.load(pickle_in)

# # Load CNN Inputs: Labels y
# pickle_in = open('CNNInputs_y.pickle','rb')
# y = pickle.load(pickle_in)

# Model Topology Definition
model = ModelTopology(X)

# Continue Training
# modelName = 'CNNPerm+64x2_1573124911'
# checkpoint_path = './Model/'+modelName+'.ckpt'
# model.load_weights(checkpoint_path)

# Train Model Topology
# Regular Train
model_top = 'Topology '+ topologyName
whole_time = time.time()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)
model.fit(x = X_train,y = y_train, batch_size = 10, epochs = 20, validation_split = 0.15, callbacks = [tensorboard,cp_callback])

# predict train
y_trainPred = model.predict(x = X_train)
# predict test
y_testPred = model.predict(x = X_test)
end_time = time.time() - whole_time

# Check Train Results
TrainDataframe = pd.DataFrame(columns=['Keq/Kpm_teo','Keq/Kpm_est','Error (%)'])
TrainDataframe['Keq/Kpm_teo'] = y_train
TrainDataframe['Keq/Kpm_est'] = y_trainPred
TrainDataframe['Error (%)'] = 100*abs(TrainDataframe['Keq/Kpm_est']-TrainDataframe['Keq/Kpm_teo'])/TrainDataframe['Keq/Kpm_teo']
# Export to csv
TrainDataframe.to_csv('./vCNN/Topologies/Train/'+caseID+'_'+model_top+'.csv',sep=';')

# Check Test Results
TopologyDataframe = pd.DataFrame(columns=['Keq/Kpm_teo','Keq/Kpm_est','Error (%)'])
TopologyDataframe['Keq/Kpm_teo'] = y_test
TopologyDataframe['Keq/Kpm_est'] = y_testPred
TopologyDataframe['Error (%)'] = 100*abs(TopologyDataframe['Keq/Kpm_est']-TopologyDataframe['Keq/Kpm_teo'])/TopologyDataframe['Keq/Kpm_teo']
# Export to csv
TopologyDataframe.to_csv('./vCNN/Topologies/Test/'+caseID+'_'+model_top+'.csv',sep=';')

print(' ')
print('Case Topology: %s' % model_top)
print('Case Batch: %s' % caseID)
print("Total time = %g [s]\n" % end_time)

'''
print('')
for i in range(0,len(y_Predicted)):
    print('{:.4f} , {:.4f} , {:.2f}, {:.2f}%'.format(\
        y_Predicted[i][0],y_test[i],100*np.abs(y_Predicted[i][0]-y_test[i])/max(y_test),100*np.abs(y_Predicted[i][0]-y_test[i])/y_test[i]))

model_eval = "print('Model evaluation ',model.evaluate(X_test,y_test)) > ./Results/" + caseId +".txt"
'''