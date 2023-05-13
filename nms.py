"""
Nazwa: nms.py
Opis: Końcowe przetwarzanie wyjść z modelu. Filtracja po pewności,
      transformacja koordynatów ramek, NMS.
Autor: Bartłomiej Moroz
"""

import torch
import torchvision


def transform_bbox(bboxes: torch.Tensor):
    """
    Transform bounding boxes from (x, y, w, h) into (x1, y1, x2, y2).
    """
    new_bboxes = bboxes.new(bboxes.shape)
    new_bboxes[..., 2:4] = bboxes[..., 2:4] / 2
    new_bboxes[..., 0:2] = bboxes[..., 0:2] - new_bboxes[..., 2:4]
    new_bboxes[..., 2:4] = bboxes[..., 0:2] + new_bboxes[..., 2:4]
    return new_bboxes


def apply_confidence_threshold(predictions: torch.Tensor, objectness_confidence: float) -> torch.Tensor:
    """
    Filter out predictions that are below objectness confidence threshold.
    """
    return predictions[predictions[..., 4] > objectness_confidence, :]


def find_best_class(predictions: torch.Tensor) -> torch.Tensor:
    """
    Find the best class for each prediction (highest class confidence),
    multiply objectness by class confidence and save best class index
    (instead of confidences of all classes).
    (x1, y1, x2, y2, objectness, classes...) -> (x1, y1, x2, y2, final_confidence, class)
    """
    # Get the most likely class for each bbox
    max_conf_val, max_conf_idx = torch.max(predictions[..., 5:], dim=1)
    max_conf_idx = max_conf_idx.unsqueeze(1)
    # Final confidence = objectness * class confidence
    predictions[..., 4] *= max_conf_val
    # Ditch all classes in bbox, instead save best class idx
    return torch.cat([predictions[..., :5], max_conf_idx], dim=1)


def non_maximum_suppression(x: torch.Tensor, iou: float) -> torch.Tensor:
    """
    Perform Non Maximum Suppression for all predictions (all classes) in an image.
    """
    # Non maximum suppression is performed per class
    classes = torch.unique(x[..., 5])
    results = x.new_empty(0, x.size(1))
    for cls in classes:
        # Get predictions containing this class
        preds_of_class = x[x[..., 5] == cls]
        # Sort by descending objectness confidence
        _, sorted_indices = torch.sort(preds_of_class[..., 4], descending=True)
        preds_of_class = preds_of_class[sorted_indices]
        # NMS proper
        result_indices = torchvision.ops.nms(preds_of_class[..., :4], preds_of_class[..., 4], iou)
        preds_of_class = preds_of_class[result_indices]
        results = torch.cat([results, preds_of_class], dim=0)
    return results


def reduce_boxes(predictions: torch.Tensor, confidence_threshold=0.3, iou=0.5, min_max_size=(2, 416)):
    """
    Given a batch of predictions, perform some transformations and reduce them to only the meaningful ones:
      - filter out low objectness
      - filter out too small or too big boxes
      - transform (x, y, w, h) boxes into (x1, y1, x2, y2)
      - find best class for each prediction
      - filter out low objectness * class_confidence
      - perform NMS
      - prepend each prediction with image index in batch
    Additionally, the batch is flattened from (batch_size, all_predictions, model_output_predictions) into (-1, final_prediction_shape).
    Final prediction shape: (id_in_batch, x1, y1, x2, y2, final_confidence, class)
    """
    all_results = predictions.new_empty(0, 7)
    # Process every image separately, NMS can't be vectorized
    for i, x in enumerate(predictions):
        # Filter out low objectness results
        x = apply_confidence_threshold(x, confidence_threshold)
        # Filter out invalid box width/height
        x = x[((x[..., 2:4] > min_max_size[0]) & (x[..., 2:4] < min_max_size[1])).all(1)]
        if x.size(0) == 0:
            continue
        # Transform bbox from (x, y, w, h, ...) into (x1, y1, x2, y2, ...)
        x[..., :4] = transform_bbox(x[..., :4])
        # Choose and save the most probable class
        x = find_best_class(x)
        # Filter out low final confidence results
        x = apply_confidence_threshold(x, confidence_threshold)
        if x.size(0) == 0:
            continue
        # NMS
        result_preds = non_maximum_suppression(x, iou)
        # Add index in batch to all image predictions
        indices_in_batch = result_preds.new(result_preds.size(0), 1).fill_(i)
        result_preds_with_indices = torch.cat([indices_in_batch, result_preds], dim=1)
        # Save results
        all_results = torch.cat([all_results, result_preds_with_indices], dim=0)
    return all_results
