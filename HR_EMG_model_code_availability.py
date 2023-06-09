# handwriting recognition - EMG model

import numpy as np
import keras
from keras.models import Sequential
from keras import layers, regularizers
from keras.utils.np_utils import to_categorical


# import datasets
x_train_aug = np.load('npy/hdw_sub2_feat_training_aug.npy')
y_train_aug = np.load('npy/hdw_sub2_label_training_aug.npy')
x_valid_aug = np.load('npy/hdw_sub2_feat_validating_aug.npy')
y_valid_aug = np.load('npy/hdw_sub2_label_validating_aug.npy')

x_train_aug = np.concatenate((x_train_aug, x_valid_aug), axis=0)
y_train_aug = np.concatenate((y_train_aug, y_valid_aug), axis=0)

x_test = np.load('npy/hdw_sub3_feat_testing.npy')
y_test = np.load('npy/hdw_sub3_label_testing.npy')
x_test_aug = np.load('npy/hdw_sub2_feat_testing_aug.npy')
y_test_aug = np.load('npy/hdw_sub2_label_testing_aug.npy')

# normalization
for i in range(x_train_aug.shape[0]):
    x_train_aug[i, :, :, :] = (x_train_aug[i, :, :, :] - np.mean(x_train_aug[i, :, :, :]))/np.std(x_train_aug[i, :, :, :])

# normalization
for i in range(x_test_aug.shape[0]):
    x_test_aug[i, :, :, :] = (x_test_aug[i, :, :, :] - np.mean(x_test_aug[i, :, :, :]))/np.std(x_test_aug[i, :, :, :])


y_train_one_hot = to_categorical(y_train_aug-1)
y_test_aug_one_hot = to_categorical(y_test_aug-1)
y_test_one_hot = to_categorical(y_test-1)
num_soft = 30

# design and train an EMG model - SepCNN
model_cnn = Sequential()
model_cnn.add(layers.SeparableConv2D(128, (3, 3), activation='relu', input_shape=(8, 33, 126)))
model_cnn.add(layers.BatchNormalization())
model_cnn.add(layers.MaxPool2D((3, 3)))
model_cnn.add(layers.Flatten())
model_cnn.add(layers.Dropout(0.5))
model_cnn.add(layers.Dense(80, kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.001), activation='relu'))
model_cnn.add(layers.BatchNormalization())
model_cnn.add(layers.Dropout(0.5))
model_cnn.add(layers.Dense(10, kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.001), activation='relu'))
model_cnn.add(layers.BatchNormalization())
model_cnn.add(layers.Dropout(0.5))
model_cnn.add(layers.Dense(num_soft, activation='softmax'))

model_cnn.compile(optimizer='RMSprop', loss='categorical_crossentropy', metrics=['acc'])
# save the EMG model
callbacks_list = [keras.callbacks.ModelCheckpoint(filepath='saves/hdw_sub2_EM' + '.h5',
                                                  monitor='val_acc', save_best_only=True)]

# start training
history_2dsepcnn = model_cnn.fit(x_train_aug, y_train_one_hot, epochs=1200, batch_size=4096, callbacks=callbacks_list,
                                 validation_data=(x_test, y_test_one_hot))


