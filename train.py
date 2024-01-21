from model.unet import *
from model.losses import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MSE
import os
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import sys
from util.data import DataGenerator
import numpy as np
import time
from pathlib import Path
import pickle

EPOCHS = 100
TARGET_SIZE = 512
BATCH_SIZE = 3
GAMMA = 7
OUT_PATH = Path(f'saved_model/{time.strftime("%Y%m%d-%H%M%S")}')


def train(epochs=EPOCHS, target_size=TARGET_SIZE, batch_size=BATCH_SIZE, out_path=OUT_PATH, gamma=GAMMA, loss_func=None, second_model=False):
    print("Start Training...")
    start = time.time()

    data_dir = f'data_{target_size}_{gamma}'

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    if second_model:
        image_folder = 'pred_distance_maps'
        mask_folder = 'masks_front_morph'
    else:
        image_folder = 'images'
        mask_folder = 'distance_maps'

    print(f'Input:  {image_folder}')
    print(f'Output: {mask_folder}')

    # create data generator
    train_generator = DataGenerator(batch_size=batch_size,
                                    train_path=f'{data_dir}/train',
                                    image_folder=image_folder,
                                    mask_folder=mask_folder)

    val_generator = DataGenerator(batch_size=batch_size,
                                  train_path=f'{data_dir}/val',
                                  image_folder=image_folder,
                                  mask_folder=mask_folder)

    # create optimizer
    if second_model:
        optimizer = Adam()  # lr=1e-5)
    else:
        optimizer = Adam()

    loss = None
    if 'dist_dice' in loss_func:
        loss = distance_weighted_dice_loss
    elif 'dist_bce' in loss_func:
        loss = distance_weighted_bce_loss
    elif 'dice' in loss_func:
        loss = dice_loss
    elif 'bce' in loss_func:
        loss = BinaryCrossentropy()
    else:
        sys.exit("Undefined Loss Function")

    # create model
    model = modern_unet(dropout=None)
    model.compile(optimizer=optimizer, loss=loss)  # cross_dice if second_model else MSE)  # metrics=[keras.metrics.MAE])
    print(model.summary())
    print(f'Loss: {loss_func}')

    callbacks = [ModelCheckpoint(f'{out_path}/unet.hdf5', monitor='val_loss', verbose=1, save_best_only=True),
                 EarlyStopping(monitor='val_loss', patience=30, verbose=1)]

    # model training
    num_samples = len(os.listdir(f'{data_dir}/train/{image_folder}'))  # number of training samples
    num_val_samples = len(os.listdir(f'{data_dir}/val/{image_folder}'))  # number of validation samples
    steps_per_epoch = np.ceil(num_samples / batch_size)
    validation_steps = np.ceil(num_val_samples / batch_size)

    history = model.fit(train_generator,
                        epochs=epochs,
                        steps_per_epoch=steps_per_epoch,
                        validation_data=val_generator,
                        validation_steps=validation_steps,
                        shuffle=True,
                        callbacks=callbacks)

    with open(f'{out_path}/history.pickle', 'wb') as f:
        pickle.dump(history.history, f)

    end = time.time()
    print('Execution Time: ', end - start)

# train()
