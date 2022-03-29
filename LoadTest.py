# Load Trained Model Test


# Pre-Processing Routines
from PreProcessing import *

# Input Parameters
from Inputs import *

# step 1 - Read Images and CSV into CNN Inputs; Features and Labels
imFiles, Perms, PMPerms = read_perm_data("Data/AM5_data.csv",delimiter=";", imlist = os.listdir("Images"))
PermIncrement = []
for i in range(0,len(Perms)):
    PermIncrement.append(Perms[i]/PMPerms[i])

# Define Training Data
training_data = create_traning_data('Images',imFiles,PermIncrement,imgSize=imgSize)

# Shuffle Training data
# random.shuffle(training_data)

X = []
y = []
# Append Features and Labels
for features, label in training_data:
    X.append(features)
    y.append(label)


X = np.array(X).reshape(-1,imgSize,imgSize,1)

X = X/255.0

model = ModelTopology(X)

modelName = 'CNNPerm+64x2_1573124911'
checkpoint_path = './Model/'+modelName+'.ckpt'



model.load_weights(checkpoint_path)

# Regular Prediction
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10)
y_Predicted = model.predict(x = X_test)

# Check Test Results
print('')
for i in range(0,len(y_Predicted)):
    print('{:.4f} , {:.4f} , {:.2f}%'.format(\
        y_Predicted[i][0],y_test[i],100*np.abs(y_Predicted[i][0]-y_test[i])/y_test[i]))

print('Model evaluation ',model.evaluate(X_test,y_test))