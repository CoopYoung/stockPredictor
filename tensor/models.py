import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, BatchNormalization

def create_advanced_model():
    ''' Initial model, changing so we can do Input(shape=(60, 7)) instead of input_shape=(60, 7)

    model = Sequential([
        LSTM(100, return_sequences=True, input_shape=(60, 7)),
        Dropout(0.2),
        LSTM(100, return_sequences=True),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    '''
    # Define input layer
    inputs = Input(shape=(60, 7))
    
    # LSTM layers
    x = LSTM(100, return_sequences=True)(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    
    x = LSTM(100, return_sequences=True)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    
    x = LSTM(50)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    
    # Dense layers
    x = Dense(25, activation='relu')(x)
    x = BatchNormalization()(x)
    outputs = Dense(1)(x)
    
    # Create model
    model = Model(inputs=inputs, outputs=outputs)
    
    # Compile model
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss=directional_loss,
                  metrics=[
                      tf.keras.metrics.MeanSquaredError(),
                      tf.keras.metrics.MeanAbsoluteError()
                  ])
    
    # Print model summary
    model.summary()

    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', 
                                                       factor=0.5, 
                                                       patience=3, 
                                                       min_lr=0.00001)
    
    return model, lr_scheduler

def directional_loss(y_true, y_pred):

    huber = tf.keras.losses.Huber()(y_true, y_pred)

    direction_true = y_true[1:] - y_true[:-1]
    direction_pred = y_pred[1:] - y_pred[:-1]
    directional_penalty = tf.reduce_mean(tf.cast(tf.sign(direction_true) != tf.sign(direction_pred), tf.float32))

    return huber + 0.1 * directional_penalty
