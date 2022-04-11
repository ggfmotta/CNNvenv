# Model Definitions

# Inputs
from Inputs import *

# Metrics
def mean_Error(y_true, y_pred):
    return K.mean(100*np.abs(y_pred-y_true)/y_true)

def max_Error(y_true, y_pred):
    return K.max(100*np.abs(y_pred-y_true)/y_true)

def accuracy(y_true, y_pred):
    return 100*(1-K.max(np.abs(y_pred-y_true)/y_true))

def ModelTopology(XInput):
    # Nework Topology
    # Initialization
    model = Sequential()

    # Layer 1: Covolutional Layer
    model.add(Conv2D(256,(5,5),input_shape = XInput.shape[1:]))
    model.add(Activation("relu"))

    # Layer 2: MaxPooling
    model.add(MaxPooling2D(pool_size = (2,2)))

    # Layer 3: Covolutional Layer
    model.add(Conv2D(128, (3,3)))
    model.add(Activation("relu"))

    # Layer 4: MaxPooling
    model.add(MaxPooling2D(pool_size = (2,2)))

    # Layer 5: Covolutional Layer
    model.add(Conv2D(64, (3,3)))
    model.add(Activation("relu"))

    # Layer 6: MaxPooling
    model.add(MaxPooling2D(pool_size = (2,2)))

    # Flatten Matrix
    model.add(Flatten())

    # Layer 7: MLP
    model.add(Dense(64))
    model.add(Activation("relu"))
    model.add(Dropout(0.15))

    # Layer 8: Output
    model.add(Dense(1, activation='linear'))

    # Optimizator
    opt = Adam(lr = learning_rate)

    # Model Compilation
    # mape = tf.keras.losses.MeanAbsolutePercentageError()
    model.compile(loss = 'mse',optimizer = opt,metrics = [mean_Error])

    return model