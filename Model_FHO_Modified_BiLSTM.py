import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Bidirectional,
    LSTM,
    Dense,
    Dropout,
    BatchNormalization,
    Attention,
    Concatenate
)
from tensorflow.keras.optimizers import Adam
from Evaluation import evaluation


# FHO (Feral Horse Optimization) Placeholder
def FHO_Optimization(train_data, train_target):

    # Replace this section with actual FHO algorithm

    best_solution = {
        'lstm_units1': 128,
        'lstm_units2': 64,
        'dense_units': 64,
        'dropout': 0.3,
        'learning_rate': 0.001,
        'batch_size': 32,
        'epochs': 50
    }

    return best_solution


# Modified BiLSTM Architecture
def build_Modified_BiLSTM(input_shape, n_classes, params):

    inputs = Input(shape=input_shape)

    # BiLSTM Layer 1
    x = Bidirectional(
        LSTM(
            params['lstm_units1'],
            return_sequences=True
        )
    )(inputs)

    x = BatchNormalization()(x)

    # BiLSTM Layer 2
    x2 = Bidirectional(
        LSTM(
            params['lstm_units2'],
            return_sequences=True
        )
    )(x)

    x2 = BatchNormalization()(x2)

    # Self Attention
    attn = Attention()([x2, x2])

    x3 = Concatenate()([x2, attn])

    # Final LSTM Compression
    x4 = Bidirectional(
        LSTM(
            params['lstm_units2'],
            return_sequences=False
        )
    )(x3)

    x4 = BatchNormalization()(x4)

    # Dense Layers
    x4 = Dropout(params['dropout'])(x4)

    x4 = Dense(
        params['dense_units'],
        activation='relu'
    )(x4)

    x4 = BatchNormalization()(x4)

    x4 = Dropout(params['dropout'])(x4)

    # Output Layer
    if n_classes == 2:

        outputs = Dense(
            1,
            activation='sigmoid'
        )(x4)

        loss_function = 'binary_crossentropy'

    else:

        outputs = Dense(
            n_classes,
            activation='softmax'
        )(x4)

        loss_function = 'sparse_categorical_crossentropy'

    model = Model(inputs, outputs)

    model.compile(
        optimizer=Adam(
            learning_rate=params['learning_rate']
        ),
        loss=loss_function,
        metrics=['accuracy']
    )

    return model


# Training Function
def Modified_BiLSTM_train(
        train_data,
        train_target,
        test_data,
        test_target, batch):

    # FHO Optimization
    params = FHO_Optimization(
        train_data,
        train_target
    )

    # Reshape for BiLSTM
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

    model = build_Modified_BiLSTM(
        (
            train_data.shape[1],
            train_data.shape[2]
        ),
        n_classes,
        params
    )

    print(model.summary())

    model.fit(
        train_data,
        train_target,
        epochs=50,
        batch_size=batch,
        validation_data=(
            test_data,
            test_target
        ),
        verbose=1
    )

    pred = model.predict(test_data)

    return pred, model


def Model_FHO_Modified_BiLSTM(
        train_data,
        train_target,
        test_data,
        test_target, batch):

    pred, model = Modified_BiLSTM_train(
        train_data,
        train_target,
        test_data,
        test_target, batch
    )

    # Binary Classification
    if pred.shape[1] == 1:

        pred_class = np.zeros(pred.shape)

        pred_class[pred >= 0.5] = 1
        pred_class[pred < 0.5] = 0

        pred_class = pred_class.flatten()

    # Multi-Class Classification
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
