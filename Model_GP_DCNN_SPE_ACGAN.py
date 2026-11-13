import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv1D, BatchNormalization,
    GlobalAveragePooling1D, Dense,
    Dropout, Concatenate
)
from tensorflow.keras.optimizers import Adam
from sklearn.utils import resample
from Evaluation import evaluation


# ==========================================================
# Self-Paced Ensemble Sampling
# ==========================================================
def self_paced_ensemble(X, y):

    classes, counts = np.unique(y, return_counts=True)

    max_count = np.max(counts)

    X_balanced = []
    y_balanced = []

    for cls in classes:
        X_cls = X[y == cls]
        y_cls = y[y == cls]

        if len(X_cls) < max_count:
            X_res, y_res = resample(
                X_cls,
                y_cls,
                replace=True,
                n_samples=max_count,
                random_state=42
            )
        else:
            X_res = X_cls
            y_res = y_cls

        X_balanced.append(X_res)
        y_balanced.append(y_res)

    X_balanced = np.vstack(X_balanced)
    y_balanced = np.hstack(y_balanced)

    idx = np.random.permutation(len(X_balanced))

    return X_balanced[idx], y_balanced[idx]


# ==========================================================
# Auxiliary Classifier GAN Placeholder
# ==========================================================
def SPE_ACGAN(train_data, train_target):

    print("SPE-ACGAN Data Balancing Started...")

    train_data_balanced, train_target_balanced = self_paced_ensemble(
        train_data,
        train_target
    )

    print("Balanced Dataset Shape :", train_data_balanced.shape)

    return train_data_balanced, train_target_balanced


# ==========================================================
# Global Pooling Dilated CNN
# ==========================================================
def build_GP_DCNN(sol, input_shape, n_classes):

    inputs = Input(shape=input_shape)

    # Dilated Block 1
    x1 = Conv1D(
        filters=64,
        kernel_size=3,
        dilation_rate=1,
        padding='same',
        activation='relu'
    )(inputs)

    x1 = BatchNormalization()(x1)

    # Dilated Block 2
    x2 = Conv1D(
        filters=64,
        kernel_size=3,
        dilation_rate=2,
        padding='same',
        activation='relu'
    )(x1)

    x2 = BatchNormalization()(x2)

    # Dilated Block 3
    x3 = Conv1D(
        filters=64,
        kernel_size=3,
        dilation_rate=4,
        padding='same',
        activation='relu'
    )(x2)

    x3 = BatchNormalization()(x3)

    # Feature Fusion
    fusion = Concatenate()([x1, x2, x3])

    # Global Pooling
    gp = GlobalAveragePooling1D()(fusion)

    fc1 = Dense(
        int(sol[0]),
        activation='relu'
    )(gp)

    fc1 = Dropout(0.3)(fc1)

    fc2 = Dense(
        64,
        activation='relu'
    )(fc1)

    fc2 = Dropout(0.2)(fc2)

    if n_classes == 2:
        outputs = Dense(
            1,
            activation='sigmoid'
        )(fc2)

        loss_fun = 'binary_crossentropy'

    else:
        outputs = Dense(
            n_classes,
            activation='softmax'
        )(fc2)

        loss_fun = 'sparse_categorical_crossentropy'

    model = Model(inputs, outputs)

    model.compile(
        optimizer=Adam(learning_rate=sol[2]),
        loss=loss_fun,
        metrics=['accuracy']
    )

    return model


# Training Function
def GP_DCNN_train(
        train_data,
        train_target,
        test_data,
        test_target,batch,
        sol=None):

    if sol is None:
        sol = [5, 5, 0.01]

    # SPE-ACGAN Balancing
    train_data, train_target = SPE_ACGAN(
        train_data,
        train_target
    )

    # Reshape for CNN
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

    model = build_GP_DCNN(sol,
        (
            train_data.shape[1],
            train_data.shape[2]
        ),
        n_classes
    )

    model.fit(
        train_data,
        train_target,
        batch_size=batch,
        epochs=int(sol[1]),
        validation_data=(
            test_data,
            test_target
        ),
        verbose=1
    )

    pred = model.predict(test_data)

    return pred, model

def Model_GP_DCNN_SPE_ACGAN(
        train_data,
        train_target,
        test_data,
        test_target,batch,
        sol=None):

    if sol is None:
        sol = [5, 5, 0.01]

    pred, model = GP_DCNN_train(
        train_data,
        train_target,
        test_data,
        test_target,batch, 
        sol
    )

    if pred.shape[1] == 1:

        pred_class = np.zeros(pred.shape)

        pred_class[pred >= 0.5] = 1
        pred_class[pred < 0.5] = 0

        pred_class = pred_class.flatten()

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
