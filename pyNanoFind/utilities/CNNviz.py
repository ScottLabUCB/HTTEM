import matplotlib.pyplot as plt
import numpy as np
from keras.models import Model
from keras.layers import *
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.losses import binary_crossentropy
from keras.models import load_model

def feat_grid(model, img_tensor, num_layers, scale = False):
    layer_outputs2 = [layer.output for layer in model.layers[:num_layers]]
    activation_model2 = Model(inputs=model.inputs,outputs = layer_outputs2)

    activations2 = activation_model2.predict(img_tensor)
    layer_names2 = []
    for layer in model.layers[:num_layers]:
        layer_names2.append(layer.name)

    images_per_row = 16

    for layer_name, layer_activation in zip(layer_names2, activations2):
        n_features = layer_activation.shape[-1]
        size = layer_activation.shape[1]

        n_cols = n_features// images_per_row
        display_grid = np.zeros((size*n_cols,images_per_row * size))

        for col in range(n_cols):
            for row in range(images_per_row):
                channel_image = layer_activation[0,:,:,col*images_per_row+row]
                channel_image -= channel_image.mean()
                channel_image /= channel_image.std()
                channel_image *= 64
                channel_image += 128
                channel_image = np.clip(channel_image,0,255).astype('uint8')
                display_grid[col*size:(col+1)*size,row*size:(row+1)*size] = channel_image

        scale = 1. / size
        plt.figure(figsize=(scale*display_grid.shape[1],scale*display_grid.shape[0]))
        plt.title(layer_name)
        plt.grid(False)
        plt.imshow(display_grid,aspect = 'auto', cmap='viridis')

def feat_activations(model,img_tensor,num_layers):
    layer_outputs = [layer.output for layer in model.layers[:num_layers]]
    activation_model = Model(inputs=model.inputs,outputs = layer_outputs)
    activations = activation_model.predict(img_tensor)
    return activations

def plotFeat(activations, layer_num, feat_num, scale = False):
    plt.figure(figsize=(20,20))
    plt.imshow(activations[layer_num][0,:,:,feat_num],cmap = 'viridis')
    if scale == True:
        plt.clim([0,1])
    plt.colorbar()

def plotFinalLayer(activations,layer_num, scale = False):
    plt.figure(figsize=(20,20))
    plt.imshow(activations[layer_num][0,:,:,:].reshape([512,512]),cmap = 'viridis')
    if scale == True:
        plt.clim([0,1])
    plt.colorbar()

def  dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)

    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)

def bce_dice_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) + dice_coef_loss(y_true, y_pred)

def down(filters, input_):
    down_ = Conv2D(filters, (3, 3), padding='same')(input_)
    down_ = BatchNormalization(epsilon=1e-4)(down_)
    down_res = Activation('relu')(down_)
    down_pool = MaxPooling2D((2, 2), strides=(2, 2))(down_)
    return down_pool, down_res

def up(filters, input_, down_):
    up_ = UpSampling2D((2, 2))(input_)
    up_ = concatenate([down_, up_], axis=3)
    up_ = Conv2D(filters, (3, 3), padding='same')(up_)
    up_ = BatchNormalization(epsilon=1e-4)(up_)
    up_ = Activation('relu')(up_)
    up_ = Dropout(0.2)(up_)
    up_ = Conv2D(filters, (3, 3), padding='same')(up_)
    up_ = BatchNormalization(epsilon=1e-4)(up_)
    up_ = Activation('relu')(up_)
    return up_

def get_unet(input_shape, num_classes=1):
    inputs = Input(shape=input_shape)

    down0, down0_res = down(4, inputs)
    down1, down1_res = down(8, down0)
    down2, down2_res = down(16, down1)
    down3, down3_res = down(32, down2)

    center = Conv2D(32, (3, 3), padding='same')(down3)
    center = BatchNormalization(epsilon=1e-4)(center)
    center = Activation('relu')(center)
    center = Conv2D(32, (3, 3), padding='same')(center)
    center = BatchNormalization(epsilon=1e-4)(center)
    center = Activation('relu')(center)

    up3 = up(32, center, down3_res)
    up2 = up(16, up3, down2_res)
    up1 = up(8, up2, down1_res)
    up0 = up(4, up1, down0_res)

    classify = Conv2D(num_classes, (1, 1), activation='sigmoid', name='final_layer')(up0)

    model = Model(inputs=inputs, outputs=classify)

    return model
