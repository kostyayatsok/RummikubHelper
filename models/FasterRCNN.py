import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    fasterrcnn_mobilenet_v3_large_320_fpn
)
from torchvision.ops import boxes as box_ops
from torchvision.models.detection.roi_heads import RoIHeads

def fastrcnn_loss(class_logits, box_regression, labels, regression_targets):
    labels = torch.cat(labels, dim=0)
    regression_targets = torch.cat(regression_targets, dim=0)
    classification_loss = 0
    values_mask = (labels < 15)
    if values_mask.sum() > 0:
        values_loss = F.cross_entropy(class_logits[values_mask,:15], labels[values_mask])
        classification_loss += values_loss
    colors_mask = (labels >= 15)
    if colors_mask.sum() > 0:
        colors_loss = F.cross_entropy(class_logits[colors_mask,15:20], labels[colors_mask]-15)
        classification_loss += colors_loss
        print(colors_loss)
    # get indices that correspond to the regression targets for
    # the corresponding ground truth labels, to be used with
    # advanced indexing
    sampled_pos_inds_subset = torch.where(labels > 0)[0]
    labels_pos = labels[sampled_pos_inds_subset]
    N, num_classes = class_logits.shape
    box_regression = box_regression.reshape(N, box_regression.size(-1) // 4, 4)

    box_loss = F.smooth_l1_loss(
        box_regression[sampled_pos_inds_subset, labels_pos],
        regression_targets[sampled_pos_inds_subset],
        beta=1 / 9,
        reduction="sum",
    )
    box_loss = box_loss / labels.numel()

    return classification_loss, box_loss


def postprocess_detections(
    self,
    class_logits,    # type: Tensor
    box_regression,  # type: Tensor
    proposals,       # type: List[Tensor]
    image_shapes     # type: List[Tuple[int, int]]
):
    # type: (...) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]
    device = class_logits.device
    num_classes = class_logits.shape[-1]

    boxes_per_image = [boxes_in_image.shape[0] for boxes_in_image in proposals]
    pred_boxes = self.box_coder.decode(box_regression, proposals)

    pred_scores = torch.cat(
        (
            F.softmax(class_logits[:,:15], -1),
            F.softmax(class_logits[:,15:20], -1)
        ),
        dim=-1
    )

    pred_boxes_list = pred_boxes.split(boxes_per_image, 0)
    pred_scores_list = pred_scores.split(boxes_per_image, 0)

    all_boxes = []
    all_scores = []
    all_labels = []
    for boxes, scores, image_shape in zip(pred_boxes_list, pred_scores_list, image_shapes):
        boxes = box_ops.clip_boxes_to_image(boxes, image_shape)

        # create labels for each prediction
        labels = torch.arange(num_classes, device=device)
        labels = labels.view(1, -1).expand_as(scores)

        
        # remove predictions with the background label
        boxes = boxes[:, 1:-1]
        scores = scores[:, 1:-1]
        labels = labels[:, 1:-1]
        
        # batch everything, by making every class prediction be a separate instance
        boxes = boxes.reshape(-1, 4)
        scores = scores.reshape(-1)
        labels = labels.reshape(-1)

        # remove low scoring boxes
        inds = torch.where(scores > self.score_thresh)[0]
        boxes, scores, labels = boxes[inds], scores[inds], labels[inds]
        
        # remove empty boxes
        keep = box_ops.remove_small_boxes(boxes, min_size=1e-2)
        boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

        # non-maximum suppression, independently done per class
        keep = box_ops.batched_nms(boxes, scores, labels, self.nms_thresh)
        # keep only topk scoring predictions
        keep = keep[:self.detections_per_img]
        boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

        all_boxes.append(boxes)
        all_scores.append(scores)
        all_labels.append(labels)

    return all_boxes, all_scores, all_labels


class FasterRCNN(nn.Module):
    def __init__(self, num_classes=20, base_model=fasterrcnn_resnet50_fpn) -> None:
        super().__init__()
        torchvision.models.detection.roi_heads.fastrcnn_loss = fastrcnn_loss
        torchvision.models.detection.roi_heads.RoIHeads.postprocess_detections = postprocess_detections
        
        self.model = base_model(pretrained=True)
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        
    def forward(self, images, targets=None):
        return self.model(images, targets)