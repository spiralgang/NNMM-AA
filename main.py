#!/usr/bin/env python3
"""
An extremely extensive and advanced TensorFlow neural network script.
This script constructs a complex model using custom layers, residual blocks,
a custom training loop, distributed strategy, and detailed logging via TensorBoard.
It is designed to push the envelope of complexity and demonstrate intricate TensorFlow mechanics.
"""

import os
import datetime
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers, initializers, activations
from tensorflow.python.ops import math_ops

# Suppress TensorFlow logging for clarity
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0 = DEBUG, 1 = INFO, 2 = WARNING, 3 = ERROR

# -----------------------------------------------------------------------------
# Custom Layer Definitions
# -----------------------------------------------------------------------------

class CustomDense(layers.Layer):
    def __init__(self,
                 units,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 **kwargs):
        super(CustomDense, self).__init__(**kwargs)
        self.units = units
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

    def build(self, input_shape):
        last_dim = int(input_shape[-1])
        self.kernel = self.add_weight(
            name='kernel',
            shape=(last_dim, self.units),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            trainable=True)
        if self.use_bias:
            self.bias = self.add_weight(
                name='bias',
                shape=(self.units,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                trainable=True)
        else:
            self.bias = None
        super(CustomDense, self).build(input_shape)

    def call(self, inputs):
        output = math_ops.matmul(inputs, self.kernel)
        if self.use_bias:
            output = math_ops.add(output, self.bias)
        if self.activation is not None:
            output = self.activation(output)
        return output

    def get_config(self):
        config = super(CustomDense, self).get_config()
        config.update({
            "units": self.units,
            "activation": activations.serialize(self.activation),
            "use_bias": self.use_bias,
            "kernel_initializer": initializers.serialize(self.kernel_initializer),
            "bias_initializer": initializers.serialize(self.bias_initializer),
            "kernel_regularizer": regularizers.serialize(self.kernel_regularizer),
            "bias_regularizer": regularizers.serialize(self.bias_regularizer),
        })
        return config


class ResidualBlock(layers.Layer):
    def __init__(self, units, activation='relu', **kwargs):
        super(ResidualBlock, self).__init__(**kwargs)
        self.units = units
        self.activation = activation
        # Two custom dense layers with L2 regularization
        self.dense_1 = CustomDense(units,
                                   activation=self.activation,
                                   kernel_regularizer=regularizers.l2(1e-4))
        self.dense_2 = CustomDense(units,
                                   activation=None,
                                   kernel_regularizer=regularizers.l2(1e-4))
        self.activation_layer = layers.Activation(self.activation)

    def call(self, inputs):
        shortcut = inputs
        x = self.dense_1(inputs)
        x = self.dense_2(x)
        x = layers.add([x, shortcut])
        return self.activation_layer(x)

    def get_config(self):
        config = super(ResidualBlock, self).get_config()
        config.update({
            "units": self.units,
            "activation": self.activation,
        })
        return config

# -----------------------------------------------------------------------------
# Complex Model Definition via Subclassing
# -----------------------------------------------------------------------------

class ComplexModel(tf.keras.Model):
    def __init__(self,
                 input_dim,
                 output_dim,
                 hidden_units=[128, 256, 128],
                 num_residual_blocks=3,
                 dropout_rate=0.25,
                 **kwargs):
        super(ComplexModel, self).__init__(**kwargs)
        # Initial dense layer
        self.input_dense = CustomDense(hidden_units[0],
                                       activation='relu',
                                       kernel_regularizer=regularizers.l2(1e-4))
        # Additional hidden dense layers
        self.hidden_layers = []
        for units in hidden_units[1:]:
            self.hidden_layers.append(CustomDense(units,
                                                  activation='relu',
                                                  kernel_regularizer=regularizers.l2(1e-4)))
        # Insert residual blocks
        self.residual_blocks = []
        for _ in range(num_residual_blocks):
            self.residual_blocks.append(ResidualBlock(hidden_units[-1], activation='relu'))
        # Batch Normalization and dropout
        self.batch_norm = layers.BatchNormalization()
        self.dropout = layers.Dropout(rate=dropout_rate)
        # Output layer with softmax activation (suitable for classification tasks)
        self.output_dense = CustomDense(output_dim,
                                        activation='softmax',
                                        kernel_regularizer=regularizers.l2(1e-4))

    def call(self, inputs, training=False):
        x = self.input_dense(inputs)
        for hidden in self.hidden_layers:
            x = hidden(x)
        for block in self.residual_blocks:
            x = block(x)
        x = self.batch_norm(x, training=training)
        x = self.dropout(x, training=training)
        return self.output_dense(x)

# -----------------------------------------------------------------------------
# Custom Callback for Learning Rate Scheduling & Logging
# -----------------------------------------------------------------------------

class CustomLearningRateScheduler(tf.keras.callbacks.Callback):
    def __init__(self, initial_lr, decay_factor, decay_steps):
        super(CustomLearningRateScheduler, self).__init__()
        self.initial_lr = initial_lr
        self.decay_factor = decay_factor
        self.decay_steps = decay_steps

    def on_epoch_begin(self, epoch, logs=None):
        new_lr = self.initial_lr * (self.decay_factor ** (epoch / self.decay_steps))
        tf.keras.backend.set_value(self.model.optimizer.lr, new_lr)
        print(f"[Epoch {epoch + 1}] Setting learning rate to: {new_lr:.6f}")

# -----------------------------------------------------------------------------
# Data Preparation: Synthetic Dataset Generation
# -----------------------------------------------------------------------------

def generate_synthetic_data(num_samples=20000, input_dim=64, num_classes=10):
    np.random.seed(42)
    X = np.random.rand(num_samples, input_dim).astype(np.float32)
    y = np.random.randint(0, num_classes, size=(num_samples,))
    y_categorical = tf.keras.utils.to_categorical(y, num_classes)
    return X, y_categorical

# -----------------------------------------------------------------------------
# Custom Training Loop with tf.GradientTape
# -----------------------------------------------------------------------------

def custom_train_loop(model, optimizer, loss_fn, dataset, epochs=20, log_interval=100):
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_acc = tf.keras.metrics.CategoricalAccuracy(name='train_acc')

    @tf.function
    def train_step(x_batch, y_batch):
        with tf.GradientTape() as tape:
            predictions = model(x_batch, training=True)
            loss_value = loss_fn(y_batch, predictions)
            loss_value += tf.add_n(model.losses)  # Include regularization losses
        gradients = tape.gradient(loss_value, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        train_loss.update_state(loss_value)
        train_acc.update_state(y_batch, predictions)

    for epoch in range(epochs):
        train_loss.reset_states()
        train_acc.reset_states()
        for step, (x_batch, y_batch) in enumerate(dataset):
            train_step(x_batch, y_batch)
            if step % log_interval == 0:
                print(f"Epoch {epoch + 1}, Step {step}, Loss: {train_loss.result().numpy():.4f}, Accuracy: {train_acc.result().numpy():.4f}")
        print(f"End of Epoch {epoch + 1}: Loss: {train_loss.result().numpy():.4f}, Accuracy: {train_acc.result().numpy():.4f}")

# -----------------------------------------------------------------------------
# Main Execution: Distributed Strategy, Model Compilation, Training, Saving
# -----------------------------------------------------------------------------

def main():
    # Hyperparameters and configuration
    INPUT_DIM = 64
    NUM_CLASSES = 10
    BATCH_SIZE = 128
    EPOCHS = 30
    INITIAL_LR = 1e-3
    DECAY_FACTOR = 0.9
    DECAY_STEPS = 5

    # Generate synthetic dataset and create tf.data.Dataset
    X, y = generate_synthetic_data(num_samples=30000, input_dim=INPUT_DIM, num_classes=NUM_CLASSES)
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    dataset = dataset.shuffle(buffer_size=10000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    # Use distributed strategy if multiple GPUs/TPUs are available
    strategy = tf.distribute.MirroredStrategy()
    print("Running with strategy:", strategy.__class__.__name__)
    with strategy.scope():
        # Instantiate the ComplexModel with multiple layers and residual blocks
        model = ComplexModel(input_dim=INPUT_DIM, output_dim=NUM_CLASSES,
                             hidden_units=[128, 256, 128],
                             num_residual_blocks=4,
                             dropout_rate=0.3)
        optimizer = tf.keras.optimizers.Adam(learning_rate=INITIAL_LR)
        loss_fn = tf.keras.losses.CategoricalCrossentropy()

        # Compile the model for high-level visualization of the computational graph
        model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])
        model.build(input_shape=(None, INPUT_DIM))
        model.summary()

    # Configure TensorBoard logging
    log_dir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    lr_scheduler_cb = CustomLearningRateScheduler(initial_lr=INITIAL_LR, decay_factor=DECAY_FACTOR, decay_steps=DECAY_STEPS)

    # Option 1: Train using model.fit with complex callbacks for visualization
    history = model.fit(dataset, epochs=EPOCHS, callbacks=[tensorboard_cb, lr_scheduler_cb])

    # Option 2: Alternatively, train using a fully custom training loop (comment/uncomment as desired)
    # custom_train_loop(model, optimizer, loss_fn, dataset, epochs=EPOCHS, log_interval=50)

    # Save the model with detailed architecture and weights
    model_path = 'complex_model_tf_extensive.h5'
    model.save(model_path)
    print(f"Model saved successfully to {model_path}")

if __name__ == "__main__":
    main()
