# Pre-Processing

# Input Parameters
from ModelDefinitions import *

def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()

def read_perm_data_pd(csv_file,delimiter=';',imlist = 'all'):

    strippedList = [os.path.splitext(x)[0] for x in imlist]
    
    # Initialize Output Lists
    imFiles = []
    Perms = []
    PMPerms = []

    # Reads CSV File with Pandas
    df = pd.read_csv(csv_file,delimiter=delimiter)
    
    HasImages = df[df['Image_file'].isin(strippedList)]
    
    with open(csv_file, "r") as f:
        reader = csv.reader(f, delimiter=delimiter)
        # Loop on Rows
        for i, row in enumerate(reader):
            if i > 0:
                colnum = 0
                picture_available = False
                # Loop on Columns
                for col in row:
                    # Image filename column = 0
                    
                    if colnum == 0:
                        correctName = col+'.tif'
                        # number = int(col[8:])
                        # if number < 1000:
                        #     correctName = col.replace('_s','_final-0')+'.tif'
                        # else:
                        #     correctName = col.replace('_s','_final-')+'.tif'
                        if imlist == 'all' or correctName in imlist:
                            
                            imFiles.append(correctName)
                            picture_available = True
                        
                    # Porous Media Permeability column = 7
                    elif colnum == 7 and picture_available:
                        if col == 'nan':
                            imFiles.pop()
                        else:
                            PMPerms.append(float(col))
                    # Final Permeability column = 8
                    elif colnum == 8 and picture_available:
                        if col == 'nan':
                            imFiles.pop()
                            PMPerms.pop()
                        else:
                            Perms.append(float(col))

                        #After reading this leave columns loop
                        break
                    # Go for next column
                    colnum += 1

    return imFiles,Perms,PMPerms

def read_perm_data(csv_file,delimiter=";", imlist = 'all'): # "Data/all_data.csv"
    
    # Initialize Output Lists
    imFiles = []
    Perms = []
    Porous = []
    PMPerms = []
    i=0
    
    # Reads CSV File
    with open(csv_file, "r") as f:
        reader = csv.reader(f, delimiter=delimiter)
        
        # Loop on Rows
        for i, row in enumerate(reader):
            if i > 0:
                colnum = 0
                picture_available = False
                # Loop on Columns
                for col in row:
                    # Image filename column = 0
                    
                    if colnum == 0:
                        correctName = col+'.tif'
                        # number = int(col[8:])
                        # if number < 1000:
                        #     correctName = col.replace('_s','_final-0')+'.tif'
                        # else:
                        #     correctName = col.replace('_s','_final-')+'.tif'
                        if imlist == 'all' or correctName in imlist:
                            
                            imFiles.append(correctName)
                            picture_available = True\
                            
                    # Porous Media Permeability column = 6
                    elif colnum == 6 and picture_available:
                        if col == 'nan' or col == '':
                            imFiles.pop()
                        else:
                            Porous.append(float(col)/100)
                        
                    # Porous Media Permeability column = 7
                    elif colnum == 7 and picture_available:
                        if col == 'nan' or col == '':
                            imFiles.pop()
                            Porous.pop()
                        else:
                            PMPerms.append(float(col))
                    # Final Permeability column = 8
                    elif colnum == 8 and picture_available:
                        if col == 'nan' or col == '':
                            imFiles.pop()
                            Porous.pop()
                            PMPerms.pop()
                        else:
                            Perms.append(float(col))

                        #After reading this leave columns loop
                        break
                    # Go for next column
                    colnum += 1

    return imFiles, Porous, Perms,PMPerms

def create_NN_data(ImDIR,imFiles,Target,Extra=[],imgSize=200,fileExtension='.tif'):
    
    training_data = []

    for i in range(0,len(imFiles)):
        #imgArray = cv2.imread(ImDIR+imFiles[i],cv2.IMREAD_GRAYSCALE) # Linux
        imgArray = cv2.imread(os.path.join(ImDIR,imFiles[i]),cv2.IMREAD_GRAYSCALE) # Windows
        # plt.imshow(imgArray,cmap='gray')
        # plt.show()
        reducArray = cv2.resize(imgArray,(imgSize,imgSize))
        
        # plt.imshow(reducArray,cmap='gray')
        # plt.show()
        training_data.append([imFiles[i], Extra[i], reducArray, float(Target[i])])
        # break

    # Shuffle Training data
    random.shuffle(training_data)

    X = []
    y = []
    Info = []
    # Append Features and Labels
    for filename, porosity, features, label, in training_data:
        X.append(features)
        y.append(label)
        Info.append([filename , porosity, label])
        
    X = np.array(X).reshape(-1,imgSize,imgSize,1)/255.0
    
    return X, y, Info

# Saving Paths
checkpoint_path = '/home/gmotta/CNN/vCNN/SavedModels/'+modelName+'/cp.ckpt'
checkpoint_dir = os.path.dirname(checkpoint_path)

# Callback definitions
save_path = '/home/gmotta/CNN/vCNN/SavedModels/'+modelName+'/'
#log_dir = '/home/gmotta/CNN/vCNN/logs/{}'.format(modelName) # Linux
#log_dir = os.path.join('logs',format(modelName),'') # Windows
es = EarlyStopping(monitor='val_loss',mode='min',verbose=1,patience=50)
#tensorboard = TensorBoard(log_dir)
checkpointer = ModelCheckpoint(filepath = save_path+'my_model.h5',save_best_only=True,verbose=1,monitor='val_loss',mode='min')