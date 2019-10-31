import tensorflow as tf

L2_LAMBDA = 1e-04


def _residual_block(x, size, dropout=False, dropout_prob=0.5, seed=None):
    residual = tf.layers.conv2d(x, filters=size, kernel_size=size, strides=2, padding='same',
                                kernel_initializer=tf.keras.initializers.he_normal(seed=seed),
                                kernel_regularizer=tf.keras.regularizers.l2(L2_LAMBDA))
    if dropout:
        residual = tf.nn.dropout(residual, dropout_prob, seed=seed)
    residual = tf.layers.batch_normalization(residual)
    residual = tf.nn.relu(residual)
    residual = tf.layers.conv2d(residual, filters=size, kernel_size=3, padding='same',
                                kernel_initializer=tf.keras.initializers.he_normal(seed=seed),
                                kernel_regularizer=tf.keras.regularizers.l2(L2_LAMBDA))
    residual = tf.layers.batch_normalization(residual)

    if dropout:
        residual = tf.nn.dropout(residual, dropout_prob, seed=seed)

    shortcut = tf.layers.conv2d(x, filters=size, kernel_size=1, strides=2, padding='same',
                          kernel_initializer=tf.keras.initializers.he_normal(seed=seed),
                          kernel_regularizer=tf.keras.regularizers.l2(L2_LAMBDA))
    shortcut = tf.layers.batch_normalization(shortcut)

    nn = tf.keras.layers.add([residual, shortcut])
    nn = tf.nn.relu(nn)

    return nn


def one_residual(x, keep_prob=0.5, seed=None):
    nn = tf.layers.conv2d(x, filters=16, kernel_size=3, strides=1, padding='valid',
                          kernel_initializer=tf.keras.initializers.he_normal(seed=seed),
                          kernel_regularizer=tf.keras.regularizers.l2(L2_LAMBDA))

    nn = tf.layers.batch_normalization(nn)

    nn = _residual_block(nn, 32, dropout_prob=keep_prob, seed=seed)
    nn = _residual_block(nn, 64, dropout_prob=keep_prob, seed=seed)

    nn = tf.layers.flatten(nn)
    return nn
