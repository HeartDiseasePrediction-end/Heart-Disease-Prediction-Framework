import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Conv1D,
    Dense,
    BatchNormalization,
    GlobalAveragePooling1D,
    GlobalMaxPooling1D,
    Multiply,
    Concatenate,
    Dropout
)
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from Evaluation import evaluation


# Efficient Channel Attention
def ECA_Block(x):

    gap = GlobalAveragePooling1D()(x)

    gap = Dense(
        1,
        activation='sigmoid'
    )(gap)

    gap = Dense(
        x.shape[-1],
        activation='sigmoid'
    )(gap)

    gap = tf.expand_dims(gap, axis=1)

    out = Multiply()([x, gap])

    return out


# Adaptive Dense Block
def Dense_Block(x, filters):

    c1 = Conv1D(
        filters,
        kernel_size=3,
        padding='same',
        activation='relu'
    )(x)

    c1 = BatchNormalization()(c1)

    c2 = Conv1D(
        filters,
        kernel_size=3,
        padding='same',
        activation='relu'
    )(c1)

    c2 = BatchNormalization()(c2)

    out = Concatenate()([x, c1, c2])

    return out


# Explainable Shallow Convolution Adaptive DenseEchoNet
def build_XSCA_DENet(sol,
        input_shape,
        n_classes):

    inputs = Input(shape=input_shape)

    # ------------------------------------
    # Shallow Convolution
    # ------------------------------------
    x = Conv1D(
        32,
        kernel_size=3,
        padding='same',
        activation='relu'
    )(inputs)

    x = BatchNormalization()(x)

    # Adaptive Dense Block
    x = Dense_Block(x, 32)

    # ECA Attention
    gap = GlobalAveragePooling1D()(x)

    att = Dense(
        x.shape[-1],
        activation='sigmoid'
    )(gap)

    att = tf.expand_dims(att, axis=1)

    x = Multiply()([x, att])

    # ------------------------------------
    # DenseEchoNet Block
    # ------------------------------------
    echo = Conv1D(
        64,
        kernel_size=1,
        activation='relu'
    )(x)

    echo = BatchNormalization()(echo)

    x = Concatenate()([x, echo])

    # ------------------------------------
    # Global Pooling
    # ------------------------------------
    x = GlobalAveragePooling1D()(x)

    x = Dense(
        int(sol[0]),
        activation='relu'
    )(x)

    x = Dropout(0.3)(x)

    if n_classes == 2:

        outputs = Dense(
            1,
            activation='sigmoid'
        )(x)

        loss_fun = 'binary_crossentropy'

    else:

        outputs = Dense(
            n_classes,
            activation='softmax'
        )(x)

        loss_fun = 'sparse_categorical_crossentropy'

    model = Model(inputs, outputs)

    model.compile(
        optimizer=Adam(sol[2]),
        loss=loss_fun,
        metrics=['accuracy']
    )

    return model


# Training
def Model_XSCA_DENet(
        train_data,
        train_target,
        test_data,
        test_target, batch, sol=None):
    if sol is None:
        sol=[5, 5, 0.01]

    if len(train_data.shape) == 2:

        train_data = train_data.reshape(
            train_data.shape[0],
            train_data.shape[1],
            1
        )

        test_data = test_data.reshape(
            test_data.shape[0],
            test_data.shape[1],
            1
        )

    n_classes = len(np.unique(train_target))

    model = build_XSCA_DENet(sol,
        (
            train_data.shape[1],
            train_data.shape[2]
        ),
        n_classes
    )

    model.fit(
        train_data,
        train_target,
        epochs=int(sol[1]),
        batch_size=batch,
        validation_data=(
            test_data,
            test_target
        ),
        verbose=1
    )

    pred = model.predict(test_data)

    if pred.shape[1] == 1:

        pred_class = (
            pred >= 0.5
        ).astype(int).flatten()

    else:

        pred_class = np.argmax(
            pred,
            axis=1
        )

    Eval = evaluation(
        pred_class,
        test_target
    )

    return Eval, pred_class