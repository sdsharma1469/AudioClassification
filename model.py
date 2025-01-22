from tensorflow.keras import layers, Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import LayerNormalization
import tensorflow as tf

# Ensure get_melspectrogram_layer is defined or imported before using this.
def get_melspectrogram_layer(input_shape, n_mels, pad_end, n_fft, win_length, hop_length, sample_rate, return_decibel, input_data_format, output_data_format):
    # This should return a mel spectrogram layer (or custom preprocessing layer).
    # Replace this with your actual implementation.
    pass

def MyConv2D(N_CLASSES=10, SR=16000, DT=1.0):
    input_shape = (int(SR * DT), 1)  # Shape of input audio samples
    
    # Generate the Mel Spectrogram as input
    i = get_melspectrogram_layer(input_shape=input_shape,
                                 n_mels=128,
                                 pad_end=True,
                                 n_fft=512,
                                 win_length=400,
                                 hop_length=160,
                                 sample_rate=SR,
                                 return_decibel=True,
                                 input_data_format='channels_last',
                                 output_data_format='channels_last')
    
    # Normalization layer
    x = LayerNormalization(axis=2, name='batch_norm')(i.output)
    
    # Convolutional and Pooling Layers
    x = layers.Conv2D(8, kernel_size=(7,7), activation='tanh', padding='same', name='conv2d_tanh')(x)
    x = layers.MaxPooling2D(pool_size=(2,2), padding='same', name='max_pool_2d_1')(x)
    
    x = layers.Conv2D(16, kernel_size=(5,5), activation='relu', padding='same', name='conv2d_relu_1')(x)
    x = layers.MaxPooling2D(pool_size=(2,2), padding='same', name='max_pool_2d_2')(x)
    
    x = layers.Conv2D(16, kernel_size=(3,3), activation='relu', padding='same', name='conv2d_relu_2')(x)
    x = layers.MaxPooling2D(pool_size=(2,2), padding='same', name='max_pool_2d_3')(x)
    
    x = layers.Conv2D(32, kernel_size=(3,3), activation='relu', padding='same', name='conv2d_relu_3')(x)
    x = layers.MaxPooling2D(pool_size=(2,2), padding='same', name='max_pool_2d_4')(x)
    
    x = layers.Conv2D(32, kernel_size=(3,3), activation='relu', padding='same', name='conv2d_relu_4')(x)
    
    # Flattening and Dense layers
    x = layers.Flatten(name='flatten')(x)
    x = layers.Dropout(rate=0.2, name='dropout')(x)
    x = layers.Dense(64, activation='relu', activity_regularizer=l2(0.001), name='dense')(x)
    
    # Output Layer
    o = layers.Dense(N_CLASSES, activation='softmax', name='softmax')(x)
    
    # Create the model
    model = Model(inputs=i.input, outputs=o, name='2d_convolution')
    
    # Compile the model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model
