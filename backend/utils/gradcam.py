import tensorflow as tf
import numpy as np
import cv2
import os

class GradCAM:
    def __init__(self, model, layer_name):
        self.model = model
        self.layer_name = layer_name
        self.grad_model = tf.keras.models.Model(
            [model.inputs], [model.get_layer(layer_name).output, model.output]
        )

    def generate_heatmap(self, img):
        if isinstance(img, np.ndarray):
            img = tf.convert_to_tensor(img, dtype=tf.float32)
        if img.ndim == 3:
            img = tf.expand_dims(img, axis=0)

        with tf.GradientTape() as tape:
            tape.watch(img)
            conv_outputs, predictions = self.grad_model(img)
            class_id = tf.argmax(predictions[0])
            class_channel = predictions[:, class_id]

        grads = tape.gradient(class_channel, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        conv_outputs = conv_outputs[0]
        pooled_grads = tf.reshape(pooled_grads, (1, 1, -1))
        pooled_grads = tf.tile(pooled_grads, [conv_outputs.shape[0], conv_outputs.shape[1], 1])

        heatmap = conv_outputs * pooled_grads
        heatmap = np.mean(heatmap, axis=-1)
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap) + 1e-8

        return heatmap

    def overlay_heatmap(self, heatmap, img_path, save_path):
        img = cv2.imread(img_path)
        heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
        superimposed_img = cv2.addWeighted(img, 0.6, heatmap_colored, 0.4, 0)

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, superimposed_img)
