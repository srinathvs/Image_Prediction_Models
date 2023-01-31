import tensorflow
from Models.resnet import resnetModelV2
from Models.vision_transformer import vision_transformer_model
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    print(str(gpu))
    tf.config.experimental.set_memory_growth(gpu, True)


class PlotLearning(tensorflow.keras.callbacks.Callback):

    def on_train_begin(self, logs={}):
        self.metrics = {}
        for metric in logs:
            self.metrics[metric] = []

    def on_epoch_end(self, epoch, logs={}):
        # Storing metrics
        for metric in logs:
            if metric in self.metrics:
                self.metrics[metric].append(logs.get(metric))
            else:
                self.metrics[metric] = [logs.get(metric)]

        # Plotting
        metrics = [x for x in logs if 'val' not in x]

        f, axs = plt.subplots(1, len(metrics), figsize=(15, 5))

        for i, metric in enumerate(metrics):
            axs[i].plot(range(1, epoch + 2),
                        self.metrics[metric],
                        label=metric)
            if logs['val_' + metric]:
                axs[i].plot(range(1, epoch + 2),
                            self.metrics['val_' + metric],
                            label='val_' + metric)

            axs[i].legend()
            axs[i].grid()

        plt.tight_layout()
        plt.show()


def normalize_resize(image, label):
    image = tf.cast(image, tf.float32)
    image = tf.divide(image, 255)
    image = tf.image.resize(image, (28, 28))
    return image, label


def augment(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_saturation(image, 0.7, 1.3)
    image = tf.image.random_contrast(image, 0.8, 1.2)
    image = tf.image.random_brightness(image, 0.1)
    return image, label


def main():
    train_ds, test_ds = tfds.load('cifar10', split=['train', 'test'], as_supervised=True)

    train = train_ds.map(normalize_resize).cache().map(augment).shuffle(100).batch(128)
    test = test_ds.map(normalize_resize).cache().batch(64)

    model_class = vision_transformer_model(input_shape=(28, 28, 3), output_shape=10, batch_size=128)
    model_predictor = model_class.predictor_model
    print(model_class.model_class_name)

    early_stopping = tensorflow.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                              patience=15,
                                                              mode='auto',
                                                              baseline=None,
                                                              restore_best_weights=True)

    history = model_predictor.fit(train, callbacks=[early_stopping, PlotLearning()], validation_data=test, epochs=200)


if __name__ == '__main__':
    main()
