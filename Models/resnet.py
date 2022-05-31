import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, GlobalAveragePooling2D, Dense, Input, BatchNormalization
from tensorflow.keras.optimizers import Adam, SGD, RMSprop


class resnetModelV2:
    def __init__(self,
                 input_shape=(512, 512, 3),
                 output_shape=10,
                 dense_layers=1,
                 use_global_pool=False,
                 use_custom_dense=True,
                 use_dense=True,
                 show_summary=True,
                 loss='mse',
                 activation='swish',
                 output_activation='sigmoid',
                 learning_rate=1e-3,
                 epsilon=1e-5,
                 optimizer='adam',
                 batch_size=128,
                 blocks_list=(2, 2, 3, 3),
                 base_filter_size=32,
                 base_dense_units=512,
                 seed=43
                 ):
        self.in_shape = input_shape
        self.out_shape = output_shape
        self.res_blocks = len(blocks_list)
        self.dense_layers = dense_layers
        self.res_pool_state = use_global_pool
        self.activation = activation
        self.output_activation = output_activation
        self.batch = batch_size
        self.dense_part = use_custom_dense
        self.res_internal_blocks = list(blocks_list)
        self.base_filter_size = base_filter_size
        self.use_dense_layers = use_dense
        self.base_dense_units = base_dense_units
        self.seed = seed
        self.model_class_name = 'resnet'
        print('Length of resnet blocks list is : ', self.res_blocks)

        self.predictor_model = self.build_model()
        if show_summary:
            self.predictor_model.summary()

        self.lr = learning_rate
        self.decay = epsilon
        self.optimizer = optimizer

        if self.optimizer == 'adam':
            optimizer_instance = Adam(
                learning_rate=self.lr,
                epsilon=self.decay
            )
        elif self.optimizer == 'sgd':
            optimizer_instance = SGD(
                learning_rate=self.lr,
                momentum=self.decay,
                nesterov=False
            )
        elif self.optimizer == 'rms':
            optimizer_instance = RMSprop(
                learning_rate=self.lr,
                centered=False,
                epsilon=1e-7,
                rho=.9
            )

        if loss == 'custom':
            self.predictor_model.compile(
                optimizer=optimizer_instance,
                loss=self.custom_loss,
                metrics=['mae', 'mse']
            )
        else:
            self.predictor_model.compile(
                optimizer=optimizer_instance,
                loss=loss,
                metrics=['mae']
            )

    def build_model(self):
        # Initial input layer to the network
        input_layer = Input(shape=self.in_shape)

        x = input_layer

        # Initial Conv block with max pooling layer
        x = Conv2D(
            filters=self.base_filter_size,
            kernel_size=14,
            strides=2,
            padding='same'
        )(x)
        x = BatchNormalization()(x)
        x = self.activation_function(x)
        x = MaxPool2D(pool_size=3, strides=2, padding='same')(x)

        # control variable for the resnet portion
        filter_size = self.base_filter_size

        # Actual residual network implementation (currently resnet v-2 with bottleneck)
        for i in range(self.res_blocks):
            if i == 0:
                for r in range(self.res_internal_blocks[i]):
                    x = self.build_identity_resnet_block(x, filter_size)
            else:
                filter_size = filter_size * 2
                x = self.build_resnet_conv_block(x, filter_size)
                if self.res_internal_blocks[i] > 1:
                    for r in range(self.res_internal_blocks[i] - 1):
                        x = self.build_identity_resnet_block(
                            x,
                            filter_size
                        )
        if not self.res_pool_state:
            x = tf.keras.layers.AveragePooling2D(
                (2, 2),
                padding='same'
            )(x)
            x = tf.keras.layers.Flatten()(x)
        else:
            x = GlobalAveragePooling2D()(x)
            x = BatchNormalization()(x)
            x = tf.nn.swish(x)

        if self.use_dense_layers:
            if not self.dense_part:
                # MLP part from the previous experiment
                x = Dense(
                    units=self.base_dense_units,
                    activation=self.output_activation
                )(x)
                x = BatchNormalization()(x)
            else:
                x = self.build_FC(x)

        x = Dense(units=self.out_shape)(x)

        output_layer = x

        model = tf.keras.models.Model(
            inputs=input_layer,
            outputs=output_layer
        )

        return model

    def build_resnet_conv_block(self, in_tensor, filters):
        shortcut = in_tensor
        x = shortcut

        # layer 1 :
        x = BatchNormalization(axis=3)(x)
        x = self.activation_function(x)
        x = Conv2D(filters, (3, 3), padding='same', strides=(2, 2))(x)

        # layer 2
        x = BatchNormalization(axis=3)(x)
        x = self.activation_function(x)
        x = Conv2D(filters, (3, 3), padding='same')(x)

        # layer 3
        x = BatchNormalization(axis=3)(x)
        x = self.activation_function(x)
        x = Conv2D(filters, (3, 3), padding='same')(x)

        # residual block addition with bottleneck
        shortcut = Conv2D(filters, (1, 1), strides=(2, 2))(shortcut)
        x = tf.keras.layers.Add()([x, shortcut])

        return x

    def build_identity_resnet_block(self, in_tensor, filters):
        shortcut = in_tensor
        x = shortcut

        # layer 1
        x = BatchNormalization(axis=-1)(x)
        x = self.activation_function(x)
        x = Conv2D(filters, (3, 3), padding='same')(x)

        # layer 2
        x = BatchNormalization(axis=3)(x)
        x = self.activation_function(x)
        x = Conv2D(filters, (3, 3), padding='same')(x)

        # layer 3
        x = BatchNormalization(axis=-1)(x)
        x = self.activation_function(x)
        x = Conv2D(filters, (3, 3), padding='same')(x)

        # Final residual addition block
        x = tf.keras.layers.Add()([x, shortcut])
        x = self.activation_function(x)
        return x

    def activation_function(self, x):
        if self.activation == 'relu':
            x = tf.nn.relu6(x)
        elif self.activation == 'swish':
            x = tf.nn.swish(x)
        elif self.activation == 'mish':
            x = mish(x)
        elif self.activation == 'leaky':
            x = tf.nn.leaky_relu(alpha=.3)
        elif self.activation == 'tanh':
            x = tf.nn.tanh(x)
        elif self.activation == 'sigmoid':
            x = tf.nn.sigmoid(x)
        elif self.activation == 'gelu':
            x = tf.nn.gelu(x)
        return x

    def custom_loss(self, y_true, y_pred):
        error = tf.abs(tf.subtract(y_true, y_pred))
        max_error = tf.reduce_max(error)
        mean_error = (
                (tf.reduce_sum(error) - max_error)
                / ((self.out_shape - 1) * self.batch)
        )

        reval_loss = (max_error + mean_error) / 2
        return reval_loss

    def save_weight(self, path='../Weights/'):
        self.predictor_model.save_weights(
            path
            + '_' + str(self.lr)
            + '_' + str(self.decay)
            + '_' + self.activation
            + '_' + str(self.dense_part)
            + '_' + str(self.res_pool_state)
            + '_' + self.optimizer,
            save_format='h5'
        )
        pass

    def build_FC(self, in_tensor):
        x = in_tensor
        x = Dense(units=1024)(x)
        x = self.activation_function(x)
        x = BatchNormalization()(x)
        x = Dense(units=512)(x)
        x = self.activation_function(x)
        x = BatchNormalization()(x)
        x = Dense(units=512)(x)
        x = self.activation_function(x)
        x = BatchNormalization()(x)
        return x


def mish(x):
    x = tf.nn.tanh(tf.nn.softplus(x))
    return x


if __name__ == '__main__':
    model_arch = resnetModelV2()
