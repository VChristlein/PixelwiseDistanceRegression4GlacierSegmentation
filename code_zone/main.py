from data import *
from model.unet import *
from model.metrics import *
from model.losses import *
from model.clr_callback import *
from model.history import append_history
from keras.utils import plot_model
import keras
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import load_model
from keras.optimizers import Adam, SGD
from pathlib import Path
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# matplotlib.use('ps')
import cv2
import os
import time
from scipy.spatial import distance
import argparse
import pickle
import sys

print(f'Keras Version: {keras.__version__}')

# %% Hyper-parameter tuning
parser = argparse.ArgumentParser(description='Glacier Front Segmentation')

# parser.add_argument('--Test_Size', default=0.2, type=float, help='Test set ratio for splitting from the whole dataset (values between 0 and 1)')
# parser.add_argument('--Validation_Size', default=0.1, type=float, help='Validation set ratio for splitting from the training set (values between 0 and 1)')

# parser.add_argument('--Classifier', default='unet', type=str, help='Classifier to use (unet/unet_Enze19)')
parser.add_argument('-e', '--epochs', default=100, type=int, help='number of training epochs (integer value > 0)')
parser.add_argument('-b', '--batch_size', default=100, type=int, help='batch size (integer value)')
parser.add_argument('-p', '--patch_size', default=256, type=int, help='patch size (integer value)')
parser.add_argument('-o', '--out_path', default=f'output/{time.time()}', type=str, help='output path for results')
parser.add_argument('-t', '--time', default=None, type=str, help='timestamp for model saving')

# parser.add_argument('--EARLY_STOPPING', default=1, type=int, help='If 1, classifier is using early stopping based on validation loss with patience 20 (0/1)')
# parser.add_argument('--Random_Seed', default=1, type=int, help='random seed number value (any integer value)')
parser.add_argument('-l', '--loss', default='binary_crossentropy', type=str, choices=['cross', 'focal', 'dice'],
                    help='loss function for the deep classifiers training (binary_crossentropy/f1_loss)')
parser.add_argument('--optimizer', default='adam', type=str, choices=['adam', 'sgd'],
                    help='optimizer for the deep classifiers training (adam/sgd)')

parser.add_argument('--clr', default=0, type=int, help='use cyclic learning rate (0/1)')
parser.add_argument('--plot_model', action="store_true", help='create model png')
parser.add_argument('--load_model', default=None, type=str, help='load model from given path')
parser.add_argument('--chained_training', default=0, type=int, help='use chained jobs for model training')
args = parser.parse_args()
# %%
START = time.time()

PATCH_SIZE = args.patch_size
batch_size = args.batch_size
load_model_path = args.load_model

# model dependencies
dependencies = {
    'dice_coef': dice_coef,
    'iou_coef': iou_coef,
    'dice_loss': dice_loss,
    'binary_focal_loss_fixed': focal_loss()
}

print(f'Epochs: {args.epochs}')
print(f'Batch Size: {args.batch_size}')
print(f'Patch Size: {args.patch_size}')
print(f'Loss function: {args.loss}')
print(f'Optimizer: {args.optimizer}')
print(f'Chained Training Iteration: {args.chained_training}')

# set base learning rate
base_lr = 1e-5

# parse loss
if 'dice' in args.loss:
    loss = dice_loss
elif 'focal' in args.loss:
    loss = focal_loss()
elif 'cross' in args.loss:
    loss = 'binary_crossentropy'
else:
    sys.exit(f"Could not find loss {args.loss}")

# parse optimizer
if 'adam' in args.optimizer:
    optimizer = Adam(lr=base_lr)
elif 'sgd' in args.optimizer:
    optimizer = SGD(lr=base_lr)

print('Loading Dataset...')
num_samples = len([file for file in Path('data_' + str(PATCH_SIZE) + '/train/images/').rglob('*.png')])  # number of training samples
num_val_samples = len([file for file in Path('data_' + str(PATCH_SIZE) + '/val/images/').rglob('*.png')])  # number of validation samples

print('Train Samples:\t' + str(num_samples))
print('Test Samples:\t' + str(num_val_samples))

# adding the time in the folder name helps to keep the results for multiple back to back executions of the code
if args.time:
    out_path = Path(f'saved_model/{args.time}_{args.loss}')
else:
    out_path = Path(f'saved_model/{time.strftime("%Y%m%d-%H%M%S")}_{args.loss}')

if not os.path.exists(out_path):
    os.makedirs(out_path)

'''
data_gen_args = dict(rotation_range=10,
                     width_shift_range=.1,
                     height_shift_range=.1,
                     shear_range=.1,
                     zoom_range=.1,
                     horizontal_flip=True,
                     vertical_flip=True,
                     fill_mode='nearest')
'''

data_gen_args = dict()

train_Generator = trainGenerator(batch_size=batch_size,
                                 train_path=f'data_{PATCH_SIZE}/train',
                                 image_folder='images',
                                 mask_folder='masks_zones',
                                 aug_dict=data_gen_args,
                                 target_size=(PATCH_SIZE, PATCH_SIZE),
                                 save_to_dir=None)

val_Generator = trainGenerator(batch_size=batch_size,
                               train_path=f'data_{PATCH_SIZE}/val',
                               image_folder='images',
                               mask_folder='masks_zones',
                               aug_dict=None,
                               target_size=(PATCH_SIZE, PATCH_SIZE),
                               save_to_dir=None)

# %%
# create model
if load_model_path:
    print("Loading saved model...")

    if not os.path.isfile(load_model_path):
        sys.exit(f'Model file does not exist: {load_model_path}')

    model = load_model(load_model_path, custom_objects=dependencies)

if os.path.isfile(f'{out_path}/training_unet.hdf5'):
    if args.chained_training:
        print("Chained Model Training...")
        model = load_model(f'{out_path}/training_unet.hdf5', custom_objects=dependencies)
    else:
        sys.exit(f'Output path already exists')
else:
    print("Compiling new model...")

    model = unet_Enze19_2(input_size=(PATCH_SIZE, PATCH_SIZE, 1))
    model.compile(optimizer=optimizer, loss=loss)  # metrics=['accuracy', dice_coef, iou_coef])

model.summary()

# %%
# plot model
if args.plot_model:
    plot_model(model, to_file=f'{out_path}/model.png', show_shapes=True, show_layer_names=True, rankdir='TB', expand_nested=False, dpi=96)
    sys.exit()

# %%
# train model
steps_per_epoch = np.ceil(num_samples / batch_size)
validation_steps = np.ceil(num_val_samples / batch_size)

callbacks = [ModelCheckpoint(f'{out_path}/unet{args.chained_training}.hdf5', monitor='val_loss', verbose=1, save_best_only=True)]

if not args.chained_training:
    print('Early Stopping: Enabled')
    callbacks.append(EarlyStopping(monitor='val_loss', patience=30, verbose=1))

# cyclic learning rate
if args.clr:
    print("Cyclic Learning Rate: Enabled")

    lr = keras.backend.get_value(model.optimizer.lr)
    base_lr = lr if args.chained_training < 2 else lr / 2
    max_lr = lr * 5

    clr_triangular = CyclicLR(mode='triangular',
                              base_lr=base_lr,
                              max_lr=max_lr,
                              step_size=steps_per_epoch * 4)  # clr authors suggest 2-8 x steps per epoch
    callbacks.append(clr_triangular)

history = model.fit_generator(train_Generator,
                              steps_per_epoch=steps_per_epoch,
                              epochs=args.epochs,
                              validation_data=val_Generator,
                              validation_steps=validation_steps,
                              shuffle=True,
                              callbacks=callbacks)

# model.fit_generator(train_Generator, steps_per_epoch=300, epochs=1, callbacks=[model_checkpoint], class_weight=[0.0000001,0.9999999])

# save model state for chained training
model.save(f'{out_path}/training_unet.hdf5')

# save model history
old_history = dict()

if os.path.exists(f'{out_path}/history.pickle'):
    with open(f'{out_path}/history.pickle', 'rb') as f:
        old_history = pickle.load(f)

history = append_history(old_history, history.history)

with open(f'{out_path}/history.pickle', 'wb') as f:
    pickle.dump(history, f)

END = time.time()
print('Execution Time: ', END - START)
