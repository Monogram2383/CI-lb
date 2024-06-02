import keras
import tensorflow as tf


class FuzzyLayer(keras.layers.Layer):
    def __init__(self, n_input: int, n_members: int, **kwargs):
        super(FuzzyLayer, self).__init__(**kwargs)
        self.n_input = n_input
        self.n_members = n_members

    def build(self, batch_size):
        self.batch_size = batch_size[0]

        self.mu = self.add_weight(name='mu',
                                  shape=(self.n_members, self.n_input),
                                  trainable=True)
        self.sigma = self.add_weight(name='sigma',
                                     shape=(self.n_members, self.n_input),
                                     trainable=True)
        super(FuzzyLayer, self).build(batch_size)

    def call(self, X):
        # gaussian
        return tf.exp(-1 * tf.square(tf.subtract(
            tf.reshape(
                tf.tile(X, (1, self.n_members)), (-1, self.n_members, self.n_input)), self.mu
        )) / tf.square(self.sigma))


class RuleLayer(keras.layers.Layer):
    def __init__(self, n_input: int, n_members: int, **kwargs):
        super(RuleLayer, self).__init__(**kwargs)
        self.n_input = n_input
        self.n_members = n_members
        self.batch_size = None

    def build(self, batch_size):
        self.batch_size = batch_size[0]
        super(RuleLayer, self).build(batch_size)

    def call(self, X):
        # правила формуються на основі системи Takagi–Sugeno–Kang
        Y = 1
        for i in range(self.n_input):
            Y *= tf.reshape(
                X[:, :, i],
                [self.batch_size, *[-1 if j == i else 1
                                    for j in range(self.n_input)]]
            )
        return tf.reshape(Y, [self.batch_size, -1])


class NormLayer(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(NormLayer, self).__init__(**kwargs)

    def call(self, w):
        w_sum = tf.reshape(tf.reduce_sum(w, axis=-1), (-1, 1))
        w_norm = w / w_sum
        return w_norm


class DefuzzLayer(keras.layers.Layer):
    def __init__(self, n_input: int, n_members: int, **kwargs):
        super(DefuzzLayer, self).__init__(**kwargs)
        self.n_input = n_input
        self.n_members = n_members

        self.consequence_bias = self.add_weight(name='Consequence_bias',
                                                shape=(1, self.n_members ** self.n_input),
                                                initializer=keras.initializers.RandomUniform(minval=-2, maxval=2),
                                                trainable=True)
        self.consequence_weight = self.add_weight(name='Consequence_weight',
                                                  shape=(self.n_input, self.n_members ** self.n_input),
                                                  initializer=keras.initializers.RandomUniform(minval=-2, maxval=2),
                                                  trainable=True)

    def call(self, w_norm, X):
        return tf.multiply(w_norm, tf.matmul(X, self.consequence_weight) + self.consequence_bias)


class SummationLayer(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(SummationLayer, self).__init__(**kwargs)
        self.batch_size = None

    def build(self, batch_size):
        self.batch_size = batch_size[0]
        super(SummationLayer, self).build(batch_size)

    def call(self, X):
        Y = tf.reduce_sum(X, axis=-1)
        return tf.reshape(Y, [-1, 1])


class ANFIS:
    def __init__(self, n_input: int, n_members: int, batch_size: int = 32):
        self.n_input = n_input
        self.n_members = n_members
        self.batch_size = batch_size

        self.input_layer = keras.layers.Input(shape=(self.n_input,), batch_size=self.batch_size)
        self.fuzzy_layer = FuzzyLayer(n_input=self.n_input, n_members=self.n_members, name='l1_fuzz')(self.input_layer)
        self.rule_layer = RuleLayer(n_input=self.n_input, n_members=self.n_members, name='l2_rule')(self.fuzzy_layer)
        self.norm_layer = NormLayer(name="l3_norm")(self.rule_layer)
        self.defuzz_layer = DefuzzLayer(n_input=self.n_input, n_members=self.n_members, name="l4_defuzz")(self.norm_layer,
                                                                                        self.input_layer)
        self.sum_layer = SummationLayer(name="l5_sum")(self.defuzz_layer)
        self.model = keras.Model(inputs=[self.input_layer], outputs=[self.sum_layer])

        self.update_weights()

    def update_weights(self):
        self.mus, self.sigmas = self.model.get_layer(name="l1_fuzz").get_weights()
        self.bias, self.weights = self.model.get_layer(name="l4_defuzz").get_weights()

    def fit(self, X, y, **kwargs):
        self.init_weights = self.model.get_layer(name="l1_fuzz").get_weights()

        history = self.model.fit(X, y, **kwargs)
        self.update_weights()

        keras.backend.clear_session()
        return history

    def compile(self, *args, **kwargs):
        self.model.compile(*args, **kwargs)


if __name__ == '__main__':
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.model_selection import train_test_split

    df = pd.read_csv("data/power_data.csv")
    df = MinMaxScaler().fit_transform(df)

    X = df[:, :-1]
    Y = df[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

    n_input = X_train.shape[1]
    n_members = 3
    batch_size = 1
    epochs=100

    model = ANFIS(
        n_input=n_input,
        n_members=n_members,
        batch_size=batch_size
    )
    model.compile(optimizer='adam', loss='mse', metrics=['mae', 'mse'])

    history = model.fit(X_train, y_train, validation_data=(X_test, y_test),epochs=epochs, batch_size=batch_size)

    # Plot the loss history
    plt.figure(figsize=(12, 5))
    plt.plot(history.history['loss'], color='r')
    plt.plot(history.history['val_loss'], color='b')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend(['train', 'val'])
    plt.show()

    # Plot the training history
    plt.figure(figsize=(12, 5))
    plt.plot(history.history['mae'], color='r')
    plt.plot(history.history['val_mae'], color='b')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.legend(['mae', 'val_mae'])
    plt.show()

    plt.figure(figsize=(12, 5))
    plt.plot(history.history['mse'], color='r')
    plt.plot(history.history['val_mse'], color='b')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.legend(['mse', 'val_mse'])
    plt.show()
