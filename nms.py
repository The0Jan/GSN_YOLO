import torch
import torchvision


def transform_bbox(bboxes: torch.Tensor):
    """
    Transform bounding boxes from (x, y, w, h) into (x1, y1, x2, y2).
    """
    new_bboxes = bboxes.new(bboxes.shape)
    for i in range(2):
        new_bboxes[..., i]   = bboxes[..., i] - bboxes[..., i+2] / 2
        new_bboxes[..., i+2] = bboxes[..., i] + bboxes[..., i+2] / 2
    return new_bboxes


def confidence_threshold(predictions: torch.Tensor, objectness_confidence: float) -> torch.Tensor:
    """
    Filter out predictions that are below objectness confidence threshold.
    """
    filtered = predictions[predictions[..., 4] > objectness_confidence]
    return filtered.view(-1, predictions.size(1))


def find_best_class(predictions: torch.Tensor) -> torch.Tensor:
    """
    """
    # Get the most likely class for each bbox
    max_conf_val, max_conf_idx = torch.max(predictions[..., 5:], dim=1)
    max_conf_idx = max_conf_idx.float().unsqueeze(1)
    max_conf_val = max_conf_val.float().unsqueeze(1)
    # Ditch all other classes in bbox, instead save class idx and class confidence
    return torch.cat([predictions[..., :5], max_conf_idx, max_conf_val], dim=1)


def non_maximum_suppression(x: torch.Tensor, iou: float) -> torch.Tensor:
    """
    Perform Non Maximum Suppression for all predictions (all classes) in an image
    """
    # Non maximum suppression is performed per class
    classes = torch.unique(x[..., 5])
    results = x.new_empty(0, 7)
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


def after_party(predictions: torch.Tensor, confidence=0.5, iou=0.5):
    """
    id_in_batch, x1, y1, x2, y2, objectness, class, class_confidence
    """
    # Transform bbox from (x, y, w, h, ...) into (x1, y1, x2, y2, ...)
    predictions[..., :4] = transform_bbox(predictions[..., :4])
    all_results = predictions.new_empty(0, 8)
    # Process every image separately, NMS can't be vectorized
    for i, x in enumerate(predictions):
        # Filter out low objectness confidence results
        x = confidence_threshold(x, confidence)
        # Choose and save the most probable class
        x = find_best_class(x)
        # NMS
        result_preds = non_maximum_suppression(x, iou)
        # Add index in batch to all image predictions
        indices_in_batch = result_preds.new(result_preds.size(0), 1).fill_(i)
        result_preds_with_indices = torch.cat([indices_in_batch, result_preds], dim=1)
        # Save results
        all_results = torch.cat([all_results, result_preds_with_indices], dim=0)
    return all_results
