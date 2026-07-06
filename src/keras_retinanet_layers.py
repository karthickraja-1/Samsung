"""
keras_retinanet_layers.py
===========================
Minimal, self-contained re-implementation of the custom Keras layers used by
the SKU110K official detector (eg4000/SKU110K_CVPR19), which is itself a fork
of fizyr/keras-retinanet with an added "IoU-Net" confidence head.

These layers are NOT part of standard Keras/TensorFlow, so they must be
registered as `custom_objects` when loading iou_resnet50_csv_XX.h5, or the
model graph cannot be reconstructed from its saved config.

Implemented here:
    - UpsampleLike        : resizes a feature map to match another (FPN)
    - PriorProbability     : bias initializer used by the classification head
    - Anchors               : generates anchor boxes for one pyramid level
    - RegressBoxes         : applies predicted deltas to anchor boxes
    - ClipBoxes            : clips boxes to the image boundary
    - FilterDetections     : score-thresholding + per-class NMS

This is a faithful, from-scratch reimplementation based on the public
fizyr/keras-retinanet design (MIT-licensed project the SKU110K repo forks),
written independently for this project so no external `keras-retinanet`
package needs to be installed.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, initializers, backend as K


# ------------------------------------------------------------------
# FPN helper
# ------------------------------------------------------------------
class UpsampleLike(layers.Layer):
    """Resizes `inputs[0]` (nearest-neighbor) to the spatial shape of `inputs[1]`."""

    def call(self, inputs, **kwargs):
        source, target = inputs
        target_shape = tf.shape(target)
        return tf.image.resize(source, (target_shape[1], target_shape[2]), method="nearest")

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0],) + input_shape[1][1:3] + (input_shape[0][-1],)


# ------------------------------------------------------------------
# Classification-head bias initializer
# ------------------------------------------------------------------
class PriorProbability(initializers.Initializer):
    """
    Initializes the final classification conv bias so that, at the start of
    training, every anchor outputs a low "foreground" probability (focal-loss
    trick from the RetinaNet paper, prevents an early loss explosion).
    """

    def __init__(self, probability: float = 0.01):
        self.probability = probability

    def __call__(self, shape, dtype=None):
        return np.ones(shape, dtype=np.float32) * -np.log((1 - self.probability) / self.probability)

    def get_config(self):
        return {"probability": self.probability}


# ------------------------------------------------------------------
# Anchor generation
# ------------------------------------------------------------------
def _generate_base_anchors(base_size: float, ratios, scales) -> np.ndarray:
    """Generate the (num_ratios * num_scales, 4) anchors centered at (0,0) for one level."""
    ratios = np.array(ratios)
    scales = np.array(scales)
    num_anchors = len(ratios) * len(scales)

    anchors = np.zeros((num_anchors, 4))
    areas = np.repeat(base_size ** 2 * scales, len(ratios))

    anchors[:, 2] = np.sqrt(areas / np.tile(ratios, len(scales)))
    anchors[:, 3] = anchors[:, 2] * np.tile(ratios, len(scales))

    anchors[:, 0] = -anchors[:, 2] / 2
    anchors[:, 1] = -anchors[:, 3] / 2
    anchors[:, 2] = anchors[:, 2] / 2
    anchors[:, 3] = anchors[:, 3] / 2
    return anchors


def _shift_anchors(feature_map_shape, stride, base_anchors):
    """Tile base_anchors over every spatial location of a feature map."""
    shift_x = (np.arange(0, feature_map_shape[1]) + 0.5) * stride
    shift_y = (np.arange(0, feature_map_shape[0]) + 0.5) * stride
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)

    shifts = np.vstack((shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel())).transpose()

    num_anchors = base_anchors.shape[0]
    num_locations = shifts.shape[0]
    all_anchors = (base_anchors.reshape((1, num_anchors, 4)) + shifts.reshape((1, num_locations, 4)).transpose((1, 0, 2)))
    return all_anchors.reshape((num_locations * num_anchors, 4)).astype(np.float32)


class Anchors(layers.Layer):
    """Generates anchor boxes for a single FPN level, matching that level's feature map size."""

    def __init__(self, size, stride, ratios=None, scales=None, **kwargs):
        self.size = size
        self.stride = stride
        self.ratios = ratios if ratios is not None else [0.5, 1, 2]
        self.scales = scales if scales is not None else [2 ** 0.0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]
        self.num_anchors = len(self.ratios) * len(self.scales)
        self.base_anchors = _generate_base_anchors(self.size, self.ratios, self.scales)
        super().__init__(**kwargs)

    def call(self, inputs, **kwargs):
        feature_shape = tf.shape(inputs)

        def _np_shift(h, w):
            return _shift_anchors((h, w), self.stride, self.base_anchors)

        anchors = tf.numpy_function(_np_shift, [feature_shape[1], feature_shape[2]], tf.float32)
        anchors = tf.reshape(anchors, (1, -1, 4))
        batch_size = feature_shape[0]
        return tf.tile(anchors, (batch_size, 1, 1))

    def compute_output_shape(self, input_shape):
        total = None if input_shape[1] is None or input_shape[2] is None else (
            input_shape[1] * input_shape[2] * self.num_anchors
        )
        return (input_shape[0], total, 4)

    def get_config(self):
        config = super().get_config()
        config.update({"size": self.size, "stride": self.stride, "ratios": self.ratios, "scales": self.scales})
        return config


# ------------------------------------------------------------------
# Box regression / clipping
# ------------------------------------------------------------------
class RegressBoxes(layers.Layer):
    """Applies predicted (dx, dy, dw, dh) deltas to anchor boxes -> absolute [x1,y1,x2,y2]."""

    def __init__(self, mean=None, std=None, **kwargs):
        self.mean = np.array(mean) if mean is not None else np.array([0, 0, 0, 0])
        self.std = np.array(std) if std is not None else np.array([0.2, 0.2, 0.2, 0.2])
        super().__init__(**kwargs)

    def call(self, inputs, **kwargs):
        anchors, deltas = inputs
        deltas = deltas * self.std + self.mean

        widths = anchors[..., 2] - anchors[..., 0]
        heights = anchors[..., 3] - anchors[..., 1]
        ctr_x = anchors[..., 0] + 0.5 * widths
        ctr_y = anchors[..., 1] + 0.5 * heights

        pred_ctr_x = ctr_x + deltas[..., 0] * widths
        pred_ctr_y = ctr_y + deltas[..., 1] * heights
        pred_w = tf.exp(deltas[..., 2]) * widths
        pred_h = tf.exp(deltas[..., 3]) * heights

        x1 = pred_ctr_x - 0.5 * pred_w
        y1 = pred_ctr_y - 0.5 * pred_h
        x2 = pred_ctr_x + 0.5 * pred_w
        y2 = pred_ctr_y + 0.5 * pred_h
        return tf.stack([x1, y1, x2, y2], axis=-1)

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def get_config(self):
        config = super().get_config()
        config.update({"mean": self.mean.tolist(), "std": self.std.tolist()})
        return config


class ClipBoxes(layers.Layer):
    """Clips boxes so they stay within the image boundary."""

    def call(self, inputs, **kwargs):
        image, boxes = inputs
        shape = tf.cast(tf.shape(image), tf.float32)
        height, width = shape[1], shape[2]

        x1 = tf.clip_by_value(boxes[..., 0], 0, width)
        y1 = tf.clip_by_value(boxes[..., 1], 0, height)
        x2 = tf.clip_by_value(boxes[..., 2], 0, width)
        y2 = tf.clip_by_value(boxes[..., 3], 0, height)
        return tf.stack([x1, y1, x2, y2], axis=-1)

    def compute_output_shape(self, input_shape):
        return input_shape[1]


# ------------------------------------------------------------------
# Detection filtering (score threshold + NMS)
# ------------------------------------------------------------------
class FilterDetections(layers.Layer):
    """
    Applies score thresholding and NMS to raw (boxes, classification[, iou])
    predictions, returning the top-K detections. If an IoU-Net confidence
    score is supplied, the final ranking score is classification * iou -
    a simplified stand-in for the SKU110K paper's full EM-based merger,
    which additionally reasons jointly over overlapping duplicate boxes.
    """

    def __init__(self, score_threshold=0.5, nms_threshold=0.45, max_detections=500, **kwargs):
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.max_detections = max_detections
        super().__init__(**kwargs)

    def call(self, inputs, **kwargs):
        if len(inputs) == 3:
            boxes, classification, iou_scores = inputs
            scores = classification[..., 0] * iou_scores[..., 0]
        else:
            boxes, classification = inputs
            scores = classification[..., 0]

        def _filter_single(args):
            boxes_i, scores_i = args
            selected = tf.image.non_max_suppression(
                boxes_i, scores_i, max_output_size=self.max_detections,
                iou_threshold=self.nms_threshold, score_threshold=self.score_threshold,
            )
            selected_boxes = tf.gather(boxes_i, selected)
            selected_scores = tf.gather(scores_i, selected)

            pad_size = tf.maximum(0, self.max_detections - tf.shape(selected_boxes)[0])
            selected_boxes = tf.pad(selected_boxes, [[0, pad_size], [0, 0]], constant_values=-1)
            selected_scores = tf.pad(selected_scores, [[0, pad_size]], constant_values=-1)
            return selected_boxes, selected_scores

        boxes_out, scores_out = tf.map_fn(
            _filter_single, (boxes, scores),
            fn_output_signature=(tf.TensorSpec(shape=(self.max_detections, 4), dtype=tf.float32),
                                  tf.TensorSpec(shape=(self.max_detections,), dtype=tf.float32)),
        )
        return [boxes_out, scores_out]

    def compute_output_shape(self, input_shape):
        return [(input_shape[0][0], self.max_detections, 4), (input_shape[0][0], self.max_detections)]

    def get_config(self):
        config = super().get_config()
        config.update({
            "score_threshold": self.score_threshold,
            "nms_threshold": self.nms_threshold,
            "max_detections": self.max_detections,
        })
        return config


# Custom objects dict for tf.keras.models.load_model(..., custom_objects=CUSTOM_OBJECTS)
CUSTOM_OBJECTS = {
    "UpsampleLike": UpsampleLike,
    "PriorProbability": PriorProbability,
    "Anchors": Anchors,
    "RegressBoxes": RegressBoxes,
    "ClipBoxes": ClipBoxes,
    "FilterDetections": FilterDetections,
}
