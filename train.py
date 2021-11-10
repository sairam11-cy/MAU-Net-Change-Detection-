import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction=0.9
session = tf.compat.v1.Session(config=config)
from keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate, BatchNormalization, Activation, add,Add,Subtract,GlobalAveragePooling2D, multiply,Reshape,Dense
from keras.models import Model, model_from_json
from keras.optimizers import Adam
from keras.layers.advanced_activations import ELU, LeakyReLU
from keras.utils.vis_utils import plot_model
from keras.callbacks import ModelCheckpoint


def conv2d_bn(x, filters, num_row, num_col, padding='same', strides=(1, 1), activation='relu', name=None):
    '''
    2D Convolutional layers
    
    Arguments:
        x {keras layer} -- input layer 
        filters {int} -- number of filters
        num_row {int} -- number of rows in filters
        num_col {int} -- number of columns in filters
    
    Keyword Arguments:
        padding {str} -- mode of padding (default: {'same'})
        strides {tuple} -- stride of convolution operation (default: {(1, 1)})
        activation {str} -- activation function (default: {'relu'})
        name {str} -- name of the layer (default: {None})
    
    Returns:
        [keras layer] -- [output layer]
    '''

    x = Conv2D(filters, (num_row, num_col), strides=strides, padding=padding, use_bias=False)(x)
    x = BatchNormalization(axis=3, scale=False)(x)

    if(activation == None):
        return x

    x = Activation(activation, name=name)(x)

    return x


def trans_conv2d_bn(x, filters, num_row, num_col, padding='same', strides=(2, 2), name=None):
    '''
    2D Transposed Convolutional layers
    
    Arguments:
        x {keras layer} -- input layer 
        filters {int} -- number of filters
        num_row {int} -- number of rows in filters
        num_col {int} -- number of columns in filters
    
    Keyword Arguments:
        padding {str} -- mode of padding (default: {'same'})
        strides {tuple} -- stride of convolution operation (default: {(2, 2)})
        name {str} -- name of the layer (default: {None})
    
    Returns:
        [keras layer] -- [output layer]
    '''

    x = Conv2DTranspose(filters, (num_row, num_col), strides=strides, padding=padding)(x)
    x = BatchNormalization(axis=3, scale=False)(x)
    
    return x


def MultiResBlock(U, inp, alpha = 1.67):
    '''
    MultiRes Block
    
    Arguments:
        U {int} -- Number of filters in a corrsponding UNet stage
        inp {keras layer} -- input layer 
    
    Returns:
        [keras layer] -- [output layer]
    '''

    W = alpha * U

    shortcut = inp

    shortcut = conv2d_bn(shortcut, int(W*0.167) + int(W*0.333) +
                         int(W*0.5), 1, 1, activation=None, padding='same')

    conv3x3 = conv2d_bn(inp, int(W*0.167), 3, 3,
                        activation='relu', padding='same')

    conv5x5 = conv2d_bn(conv3x3, int(W*0.333), 3, 3,
                        activation='relu', padding='same')

    conv7x7 = conv2d_bn(conv5x5, int(W*0.5), 3, 3,
                        activation='relu', padding='same')

    out = concatenate([conv3x3, conv5x5, conv7x7], axis=3)
    out = BatchNormalization(axis=3)(out)

    out = add([shortcut, out])
    out = Activation('relu')(out)
    out = BatchNormalization(axis=3)(out)

    return out


def ResPath(filters, length, inp):
    '''
    ResPath
    
    Arguments:
        filters {int} -- [description]
        length {int} -- length of ResPath
        inp {keras layer} -- input layer 
    
    Returns:
        [keras layer] -- [output layer]
    '''


    shortcut = inp
    shortcut = conv2d_bn(shortcut, filters, 1, 1,
                         activation=None, padding='same')

    out = conv2d_bn(inp, filters, 3, 3, activation='relu', padding='same')

    out = add([shortcut, out])
    out = Activation('relu')(out)
    out = BatchNormalization(axis=3)(out)

    for i in range(length-1):

        shortcut = out
        shortcut = conv2d_bn(shortcut, filters, 1, 1,
                             activation=None, padding='same')

        out = conv2d_bn(out, filters, 3, 3, activation='relu', padding='same')

        out = add([shortcut, out])
        out = Activation('relu')(out)
        out = BatchNormalization(axis=3)(out)

    return out

def squeeze_excitation_layer(x, out_dim):
    ratio=4
    squeeze = GlobalAveragePooling2D()(x)
    excitation = Dense(units=out_dim // ratio)(squeeze)
    excitation = Activation('relu')(excitation)
    excitation = Dense(units=out_dim)(excitation)
    excitation = Activation('sigmoid')(excitation)
    excitation = Reshape((1,1,out_dim))(excitation)
    scale = multiply([x,excitation])
    return scale
def MultiResUnet(height=256, width=256, n_channels=3):
    '''
    MultiResUNet
    
    Arguments:
        height {int} -- height of image 
        width {int} -- width of image 
        n_channels {int} -- number of channels in image
    
    Returns:
        [keras model] -- MultiResUNet model
    '''


    inputs = Input((None, None, n_channels))
    #inputs1 = Input((height, width, n_channels))

    mresblock1 = MultiResBlock(32, inputs)
    pool1 = MaxPooling2D(pool_size=(2, 2))(mresblock1)

    pool1=squeeze_excitation_layer(pool1,51)
    mresblock1 = ResPath(32, 4, mresblock1)
    
    #mresblock11 = MultiResBlock(32, inputs1)
    #pool11 = MaxPooling2D(pool_size=(2, 2))(mresblock11)
    #mresblock11 = ResPath(32, 4, mresblock11)

    mresblock2 = MultiResBlock(32*2, pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(mresblock2)
    pool2=squeeze_excitation_layer(pool2,105)
    mresblock2 = ResPath(32*2, 3, mresblock2)

    #mresblock21 = MultiResBlock(32*2, pool11)
    #pool21 = MaxPooling2D(pool_size=(2, 2))(mresblock21)
    #mresblock21 = ResPath(32*2, 3, mresblock21)

    mresblock3 = MultiResBlock(32*4, pool2)
    pool3 = MaxPooling2D(pool_size=(2, 2))(mresblock3)
    pool3=squeeze_excitation_layer(pool3,212)
    mresblock3 = ResPath(32*4, 2, mresblock3)

    #mresblock31 = MultiResBlock(32*4, pool21)
    #pool31 = MaxPooling2D(pool_size=(2, 2))(mresblock31)
    #mresblock31 = ResPath(32*4, 2, mresblock31)

    mresblock4 = MultiResBlock(32*8, pool3)
    pool4 = MaxPooling2D(pool_size=(2, 2))(mresblock4)
    pool4=squeeze_excitation_layer(pool4,426)
    mresblock4 = ResPath(32*8, 1, mresblock4)

    #mresblock41 = MultiResBlock(32*8, pool31)
    #pool41 = MaxPooling2D(pool_size=(2, 2))(mresblock41)
    #mresblock41 = ResPath(32*8, 1, mresblock41)

    mresblock5 = MultiResBlock(32*16, pool4)
    mresblock5=squeeze_excitation_layer(mresblock5,853)
    #mresblock51 = MultiResBlock(32*16, pool41)
    
    #mresblock5=Subtract()([mresblock5,mresblock51])
    up6 = concatenate([Conv2DTranspose(
        32*8, (2, 2), strides=(2, 2), padding='same')(mresblock5), mresblock4], axis=3)
    mresblock6 = MultiResBlock(32*8, up6)
    mresblock6=squeeze_excitation_layer(mresblock6,426)
    #mresblock3=Subtract()([mresblock3,mresblock31])
    
    up7 = concatenate([Conv2DTranspose(
        32*4, (2, 2), strides=(2, 2), padding='same')(mresblock6), mresblock3], axis=3)
    mresblock7 = MultiResBlock(32*4, up7)
    mresblock7=squeeze_excitation_layer(mresblock7,212)
    #mresblock2=Subtract()([mresblock2,mresblock21])
    up8 = concatenate([Conv2DTranspose(
        32*2, (2, 2), strides=(2, 2), padding='same')(mresblock7), mresblock2], axis=3)
    mresblock8 = MultiResBlock(32*2, up8)
    mresblock8=squeeze_excitation_layer(mresblock8,105)
    #mresblock1=Subtract()([mresblock1,mresblock11])
    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(
        2, 2), padding='same')(mresblock8), mresblock1], axis=3)
    mresblock9 = MultiResBlock(32, up9)
    mresblock9=squeeze_excitation_layer(mresblock9,51)
    conv10 = conv2d_bn(mresblock9, 1, 1, 1, activation='sigmoid')
    
    model = Model(inputs=[inputs], outputs=[conv10])

    return model
import os
train1=[]
train2=[]
output=[]
for filename in os.listdir('dataset/img1'):
    img=cv2.imread('dataset/img1/'+filename)
    #img=cv2.resize(img,(256,256))
    img=img/255.
    train1.append(img)
for filename in os.listdir('dataset/img1'):
    img=cv2.imread('dataset/img2/'+filename)
    #img=cv2.resize(img,(256,256))
    img=img/255.
    train2.append(img)
for filename in os.listdir('dataset/img1'):
    img=cv2.imread('dataset/mask/'+filename,0)
    #img=cv2.resize(img,(256,256))
    img=img/255.
    output.append(img)
train1=np.array(train1)
train2=np.array(train2)
img=train1-train2
print(img.shape)
output=np.array(output)
output=np.expand_dims(output,3)
model=MultiResUnet()
model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])
name="Changemodel.hdf5"
checkpoint = ModelCheckpoint(name, monitor='val_loss', save_weights_only=True,verbose=1,save_best_only=True, mode='min', period=1)
model.fit(img,output,batch_size=50,epochs=500,validation_split=0.20,callbacks=[checkpoint])
