import torch
import torchvision


def transform_bbox(bboxes):
    """
    Transform bounding boxes from (x, y, w, h) into (x1, y1, x2, y2)
    """
    new_bboxes = bboxes.new(bboxes.shape)
    for i in range(2):
        new_bboxes[..., i]   = bboxes[..., i] - bboxes[..., i+2] / 2
        new_bboxes[..., i+2] = bboxes[..., i] + bboxes[..., i+2] / 2
    return new_bboxes


def non_maximum_suppression(predictions, objectness_confidence=0.3, class_confidence=0.3 iou=0.5):
    output = [torch.zeros((0, 6), device='cpu')] * predictions.size(0)
    for i, x in enumerate(predictions):
        # filter out low objectness confidence results
        x = x[x[..., 4] > objectness_confidence]
        if not x.shape(0):
            continue
        # confidence = objectness * class_predictions
        x[..., 5:] *= x[..., 4:5]
        # Transform bbox from (x, y, w, h) into (x1, y1, x2, y2)
        bbox = transform_bbox(x[..., :4])


        i, j = (x[..., 5:] > class_confidence).nonzero(as_tuple=False).T
        x = torch.cat((bbox[i], x[i, j + 5, None], j[..., None].float()), 1)

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]


        result_indices = torchvision.ops.nms(bboxes, scores, iou)

    return 0
