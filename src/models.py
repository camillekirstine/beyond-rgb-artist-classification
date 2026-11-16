import tensorflow as tf

from src import config


# --------------------------------------------------
# CNN architecture
# --------------------------------------------------

def build_cnn():
    """
    Build the shared CNN architecture used in all experiments.

    Returns
    -------
    tf.keras.Model
        Uncompiled CNN classification model.
    """

    model = tf.keras.Sequential([

        tf.keras.layers.Input(
            shape=(
                config.IMAGE_SIZE[0],
                config.IMAGE_SIZE[1],
                config.IMAGE_CHANNELS
            )
        ),

        # Convolutional block 1
        tf.keras.layers.Conv2D(
            32,
            kernel_size=(3, 3),
            activation="relu",
            padding="same"
        ),

        tf.keras.layers.MaxPooling2D(
            pool_size=(2, 2)
        ),

        # Convolutional block 2
        tf.keras.layers.Conv2D(
            64,
            kernel_size=(3, 3),
            activation="relu",
            padding="same"
        ),

        tf.keras.layers.MaxPooling2D(
            pool_size=(2, 2)
        ),

        # Convolutional block 3
        tf.keras.layers.Conv2D(
            128,
            kernel_size=(3, 3),
            activation="relu",
            padding="same"
        ),

        tf.keras.layers.MaxPooling2D(
            pool_size=(2, 2)
        ),

        # Classification head
        tf.keras.layers.GlobalAveragePooling2D(),

        tf.keras.layers.Dense(
            128,
            activation="relu"
        ),

        tf.keras.layers.Dropout(
            config.DROPOUT_RATE
        ),

        tf.keras.layers.Dense(
            config.NUM_CLASSES,
            activation="softmax"
        )
    ])

    return model


# --------------------------------------------------
# Model compilation
# --------------------------------------------------

def compile_cnn(model):
    """
    Compile the CNN using the project's standard configuration.

    Parameters
    ----------
    model : tf.keras.Model
        CNN model to compile.

    Returns
    -------
    tf.keras.Model
        Compiled CNN model.
    """

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=config.LEARNING_RATE
        ),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model
