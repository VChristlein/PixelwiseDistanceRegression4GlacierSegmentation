from tensorflow.keras.models import *
from tensorflow.keras.layers import *


def conv_2d_block(inputs, filters, kernel_size=5, alpha=.1, strides=(1, 1), dropout=None):
    x = Conv2D(filters, kernel_size, padding='same', strides=strides, kernel_initializer='he_normal')(inputs)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha)(x)

    if dropout:
        x = SpatialDropout2D(dropout)(x)

    x = Conv2D(filters, kernel_size, padding='same', strides=strides, kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha)(x)
    return x


def conv_2d_transpose(inputs, filters, kernel_size=4, strides=(2, 2)):
    return Conv2DTranspose(filters, kernel_size, strides=strides, padding='same', kernel_initializer='he_normal')(inputs)


def modern_unet(pretrained_weights=None, input_size=(512, 512, 1), filters=32, dropout=.3, num_layers=5):
    inputs = Input(input_size)
    x = inputs
    down_layers = []

    # down sample
    for i in range(num_layers):
        x = conv_2d_block(inputs=x, filters=filters, dropout=dropout)
        down_layers.append(x)

        x = MaxPooling2D(pool_size=(2, 2))(x)

        filters *= 2

    # mid layer
    x = conv_2d_block(inputs=x, filters=filters, dropout=dropout)

    # upsample
    for conv in reversed(down_layers):
        filters //= 2

        x = conv_2d_transpose(inputs=x, filters=filters)

        x = Concatenate(axis=-1)([x, conv])
        x = conv_2d_block(inputs=x, filters=filters, dropout=dropout)

    # output layer
    outputs = Conv2D(1, 3, padding='same', activation='sigmoid', kernel_initializer='he_normal')(x)

    model = Model(inputs=inputs, outputs=outputs)

    if pretrained_weights:
        model.load_weights(pretrained_weights)

    return model


# no pooling instead strides and elu
def modern_unet2(pretrained_weights=None, input_size=(512, 512, 1), filters=32, dropout=.3, num_layers=5):
    inputs = Input(input_size)
    x = inputs
    down_layers = []

    # down sample
    for i in range(num_layers):
        x = Conv2D(filters, 5, padding='same', kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)
        x = ELU()(x)
        down_layers.append(x)

        x = Conv2D(filters, 5, padding='same', strides=(2, 2), kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)
        x = ELU()(x)

        if dropout:
            x = SpatialDropout2D(dropout)(x)

        filters *= 2

    # mid layer
    x = Conv2D(filters, 5, padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = ELU()(x)

    x = Conv2D(filters, 5, padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = ELU()(x)

    # upsample
    for conv in reversed(down_layers):
        filters //= 2

        x = conv_2d_transpose(inputs=x, filters=filters)
        x = Concatenate(axis=-1)([x, conv])

        x = Conv2D(filters, 5, padding='same', kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)
        x = ELU()(x)

        x = Conv2D(filters, 5, padding='same', kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)
        x = ELU()(x)

        if dropout:
            x = SpatialDropout2D(dropout)(x)

    # output layer
    outputs = Conv2D(1, kernel_size=(3, 3), padding='same', activation='sigmoid')(x)

    model = Model(input=inputs, output=outputs)

    if pretrained_weights:
        model.load_weights(pretrained_weights)

    return model


## kernel sizes from 5 -> 3
def modern_unet3(pretrained_weights=None, input_size=(512, 512, 1), filters=32, dropout=.3, num_layers=5):
    inputs = Input(input_size)
    x = inputs
    down_layers = []

    # down sample
    for i in range(num_layers):
        x = conv_2d_block(inputs=x, filters=filters, kernel_size=3, dropout=dropout)
        down_layers.append(x)

        x = MaxPooling2D(pool_size=(2, 2))(x)

        filters *= 2

    # mid layer
    x = conv_2d_block(inputs=x, filters=filters, kernel_size=3, dropout=dropout)

    # upsample
    for conv in reversed(down_layers):
        filters //= 2

        x = conv_2d_transpose(inputs=x, filters=filters, kernel_size=3, strides=(2, 2))

        x = Concatenate(axis=-1)([x, conv])
        x = conv_2d_block(inputs=x, filters=filters, kernel_size=3, dropout=dropout)

    # output layer
    outputs = Conv2D(1, 1, padding='same', activation='sigmoid')(x)

    model = Model(input=inputs, output=outputs)

    if pretrained_weights:
        model.load_weights(pretrained_weights)

    return model
