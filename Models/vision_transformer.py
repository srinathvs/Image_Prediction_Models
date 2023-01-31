# Based on the tensorflow keras blog post on vision transformers

import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.layers import Add, Dense, Input, MultiHeadAttention, LayerNormalization
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.layers import BatchNormalization


class vision_transformer_model:
    def __init__(self,
                 input_shape=(512, 512, 3),
                 loss='mse',
                 activation='gelu',
                 learning_rate=1e-4,
                 optimizer='adamw',
                 decay=1e-6,
                 attention_layers=16,
                 patch_size=4,
                 num_heads=6,
                 proj_dims=64,
                 key_dims=64,
                 output_activation='softmax',
                 dropout=.4,
                 batch_size=128,
                 output_shape=10,
                 show_summary=True,
                 custom_loss_func=None,

                 ):
        self.in_shape = input_shape
        self.out_shape = output_shape
        self.attention_layers = attention_layers
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.proj_dims = proj_dims
        self.key_dims = key_dims
        self.dropout = dropout
        self.output_activation = output_activation
        self.activation = activation
        self.decay = decay
        self.batch_size = batch_size
        self.loss = loss
        self.lr = learning_rate
        self.optimizer = optimizer
        self.model_class_name = 'transformer'

        if self.optimizer == 'adam':
            optimizer_instance = Adam(
                learning_rate=self.lr,
                epsilon=self.decay
            )

        elif self.optimizer == 'rms':
            optimizer_instance = RMSprop(
                learning_rate=self.lr,
                centered=False,
                epsilon=1e-7,
                rho=.9
            )
        elif self.optimizer == 'adamw':
            optimizer_instance = tfa.optimizers.AdamW(
                learning_rate=self.lr,
                weight_decay=self.decay
            )

        self.predictor_model = self.build_model()
        if show_summary:
            self.predictor_model.summary()

        self.predictor_model.compile(
            optimizer=optimizer_instance,
            loss=self.loss,
            metrics=['mae'],
        )

    def build_model(self):
        # Initial input layer to the network
        input_layer = Input(shape=self.in_shape)

        x = input_layer

        # attention_weights_list = []

        if self.attention_layers > 0:
            patch_size = self.patch_size
            num_patches = (self.in_shape[0] // patch_size) * (self.in_shape[1] // patch_size)
            patches = Patches(patch_size)(x)
            x = PatchEncoder(
                num_patches=num_patches,
                projection_dim=self.proj_dims,
                activation=None)(patches)
            # x = patches

            for _ in range(self.attention_layers):
                x = self.buildAttentionBlock(x)

        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dropout(self.dropout)(x)
        x = Dense(units=self.out_shape)(x)

        output_layer = x

        model = tf.keras.models.Model(
            inputs=input_layer,
            outputs=output_layer,
            name='AttentionModel'
        )

        return model

    def buildAttentionBlock(self, x):
        shortcut = x

        # x, attention_weights = MultiHeadAttention(num_heads=self.num_heads, key_dim=self.key_dims)(x, x, return_attention_scores=True)
        x = MultiHeadAttention(num_heads=self.num_heads, key_dim=self.key_dims)(x, x)
        x = Add()([x, shortcut])
        x = LayerNormalization(epsilon=self.decay)(x)
        x = Dense(self.proj_dims, activation=self.activation)(x)

        return x

    def customLoss(sv_pred, attn_weights, y):

        mse_loss = tf.keras.losses.MeanSquaredError(sv_pred, y)
        mae_loss = tf.keras.losses.MeanAbsoluteError(sv_pred, y)

        return mse_loss + mae_loss

    def build_FC(self, in_tensor):
        x = in_tensor
        x = Dense(units=2048)(x)
        x = self.activation_function(x)
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


class Patches(tf.keras.layers.Layer):
    def __init__(self, patch_size, **kwargs):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        num_patches = patches.shape[-2] * patches.shape[-3]
        patches = tf.reshape(patches, [batch_size, num_patches, patch_dims])

        return patches

    def get_config(self):
        cfg = super().get_config()
        return cfg


class PatchEncoder(tf.keras.layers.Layer):
    def __init__(self, num_patches, projection_dim, activation=None, **kwargs):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = tf.keras.layers.Dense(units=projection_dim, activation=activation)
        self.position_embedding = tf.keras.layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded

    def get_config(self):
        cfg = super().get_config()
        return cfg


def mish(x):
    x = tf.nn.tanh(tf.nn.softplus(x))
    return x


if __name__ == '__main__':
    transformer = vision_transformer_model()
