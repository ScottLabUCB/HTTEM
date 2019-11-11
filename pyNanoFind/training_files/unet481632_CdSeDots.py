from keras.models import Model
from keras.layers import *
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.losses import binary_crossentropy
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import h5py
import os



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

X = h5py.File('/global/scratch/cgroschner/combined_data/Bal_MedFilt_CdSeRelabel512Images_20190726.h5','r')['images'][:,:,:,:]
Y = h5py.File('/global/scratch/cgroschner/combined_data/Bal_unFilt_CdSeRelabel512Images_20190724_maps.h5','r')['maps'][:,:,:,:]
X = X/X.max()
Y = Y/Y.max()
trainX = X[:129,:,:,:]
trainY = Y[:129,:,:,:]


data_gen_args = dict(rotation_range=360,fill_mode = 'wrap',horizontal_flip = True, vertical_flip = True,validation_split=0.25)
image_datagen = ImageDataGenerator(**data_gen_args)
mask_datagen = ImageDataGenerator(**data_gen_args)
seed = 42
image_datagen.fit(trainX, augment=True, seed=seed)
mask_datagen.fit(trainY, augment=True, seed=seed)

image_generator_train = image_datagen.flow(trainX,batch_size = 20,seed = seed,subset='training')
mask_generator_train = mask_datagen.flow(trainY,batch_size = 20, seed = seed,subset='training')
image_generator_test = image_datagen.flow(trainX,batch_size = 20,seed = seed,subset='validation')
mask_generator_test = mask_datagen.flow(trainY,batch_size = 20, seed = seed,subset='validation')

train_generator = zip(image_generator_train, mask_generator_train)
test_generator = zip(image_generator_test, mask_generator_test)

save_weights = '/global/scratch/cgroschner/unet481632_CdSeDots_08.h5'
save_weights_final = '/global/scratch/cgroschner/unet481632_CdSeDots_08_final.h5'
save_predictions = '/global/scratch/cgroschner/unet481632_CdSeDots_08.npy'

if os.path.isfile(save_weights) == True:
    raise(RuntimeError('FILE ALREADY EXISTS RENAME WEIGHT FILE'))
if os.path.isfile(save_predictions) == True:
    raise(RuntimeError('FILE ALREADY EXISTS RENAME PREDICTIONS FILE'))

earlyStopping = EarlyStopping(monitor='val_loss',
                              patience=2,
                              verbose=2,
                              min_delta = 0.001,
                              mode='min',)

modelCheckpoint = ModelCheckpoint(save_weights,
                                  monitor = 'val_loss',
                                  save_best_only = True,
                                  mode = 'min',
                                  verbose = 2,
                                  save_weights_only = True)
callbacks_list = [modelCheckpoint,earlyStopping]

model = get_unet((512,512,1),2)
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=1e-4), metrics=[dice_coef])
print(model.summary())
model.fit_generator(train_generator,steps_per_epoch=1000,epochs=10,callbacks = callbacks_list,validation_data=test_generator,validation_steps=500,verbose = 2)
model.save_weights(save_weights_final)
predY = model.predict_generator(test_generator,steps=2,verbose=1)
np.save(save_predictions, predY)
validation_score = model.evaluate_generator(test_generator,steps=5)
print(validation_score)
