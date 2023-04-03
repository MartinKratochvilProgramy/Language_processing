import tensorflow as tf

def create_model(train_dataset, test_dataset):
    # vectorize the dataset
    VOCAB_SIZE = 5000
    encoder = tf.keras.layers.TextVectorization(max_tokens=VOCAB_SIZE)
    encoder.adapt(train_dataset.map(lambda text, label: text))

    # Building th RNN
    # cell = tf.keras.layers.GRUCell(64, recurrent_activation='sigmoid')
    model = tf.keras.Sequential([
        encoder,
        tf.keras.layers.Embedding(
            input_dim=len(encoder.get_vocabulary()),
            output_dim=128,
            ),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True)),
        tf.keras.layers.GlobalMaxPool1D(),
        tf.keras.layers.Dense(64, activation='sigmoid'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    # compile model
    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(),      # from_logits=True
        optimizer=tf.keras.optimizers.Adam(0.0001),
        metrics=[
            'mse'
        ]
    )

    return model
