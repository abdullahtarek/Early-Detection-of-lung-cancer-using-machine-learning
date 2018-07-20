import tensorflow as tf
import keras as k
from keras.models import Sequential ,Model
from keras.layers import Conv2D, MaxPooling2D,AveragePooling3D,AveragePooling2D,MaxPooling3D,Conv3D
from keras.layers import Activation, Dropout, Flatten, Dense ,Input
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.preprocessing import image
from keras import applications

def inceptionlayer(prev):
    tower_1 = Conv3D(64, (1,1,1), padding='same', activation='relu')(prev)
    tower_1 = Conv3D(64, (3,3,3), padding='same', activation='relu')(tower_1)
    tower_2 = Conv3D(64, (1,1,1), padding='same', activation='relu')(prev)
    tower_2 = Conv3D(64, (5,5,5), padding='same', activation='relu')(tower_2)
    tower_3 = MaxPooling3D((3,3,3), strides=(1,1,1), padding='same')(prev)
    tower_3 = Conv3D(64, (1,1,1), padding='same', activation='relu')(tower_3)
    output = k.layers.concatenate([tower_1, tower_2, tower_3], axis = 3)
    
    return output


#GOOGLENET
def makegooglenet():
    x    = Input(shape=(64,64,64,1))
    conv1= Conv3D(32,kernel_size=(7,7,7),activation='relu')(x)
    conv1= Dropout(0.3)(conv1)
    max1 = MaxPooling3D(pool_size=(2,2,2))(conv1)
    conv2= Conv3D(32,kernel_size=(3,3,3),activation='relu')(max1)
    conv2= Dropout(0.3)(conv2)
    max2 = MaxPooling3D(pool_size=(2,2,2))(conv2)
    incp1= inceptionlayer(max2)
    incp1= Dropout(0.3)(incp1)
    incp2= inceptionlayer(incp1)
    incp2= Dropout(0.3)(incp2)
    max3 = MaxPooling3D(pool_size=(2,2,2))(incp2)
    incp3= inceptionlayer(max3)
    incp3= Dropout(0.3)(incp3)
    incp4= inceptionlayer(incp3)
    incp4= Dropout(0.3)(incp4)
    max4 = MaxPooling3D(pool_size=(2,2,2))(incp4)
    incp5= inceptionlayer(max4)
    incp5= Dropout(0.3)(incp5)
    incp6= inceptionlayer(incp5)
    incp6= Dropout(0.3)(incp6)
    avg1= AveragePooling3D(pool_size=(2,2,2))(incp4)

    flat = Flatten()(avg1)
    flat= Dropout(0.3)(flat)
    dense= Dense(2,activation="softmax")(flat)

    googlenet = Model(inputs=x, outputs=dense)
    
    googlenet.compile(loss=k.losses.categorical_crossentropy,
              optimizer=k.optimizers.Adam(lr=0.001),
              metrics=['accuracy'])
    
    googlenet.load_weights('weights3d_googlenet.hdf5')
    
    return googlenet
    

    
def makelenet():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
                     activation='relu',
                     input_shape=[64,64,64]))
    model.add(Dropout(0.4))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(64, (5, 5), activation='relu'))
    model.add(Dropout(0.4))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dropout(0.4))
    model.add(Dense(1000, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(6, activation='softmax'))
    model.compile(loss=k.losses.categorical_crossentropy,
                  optimizer=k.optimizers.SGD(lr=0.01),
                  metrics=['accuracy'])
    model_checkpoint = ModelCheckpoint('malignancy_crop.hdf5', monitor='loss', save_best_only=True)
    model.load_weights("malignancy_crop.hdf5")
    return model