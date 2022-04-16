import numpy as np
import wandb
import torch
from tqdm import tqdm
from torchvision.ops.boxes import box_iou
from collections import defaultdict

def positives_and_negatives(ground_truth, predictions, iou_threshold=0.5):
    positives = defaultdict(list)
    negatives = defaultdict(int)

    for i in range(len(predictions)):
        for pred_idx in range(len(predictions[i]["boxes"])):
            used = False
            for true_idx in range(len(ground_truth[i]["boxes"])):            
                iou = box_iou(
                    predictions[i]["boxes"][pred_idx].unsqueeze(0),
                    ground_truth[i]["boxes"][true_idx].unsqueeze(0)
                )[0]
                correct = ground_truth[i]["labels"][true_idx] == \
                                            predictions[i]["labels"][pred_idx]
                if iou >= iou_threshold:
                    used = True
                    
                    positives["iou"].append(iou)
                    positives["score"].append(predictions[i]["scores"][pred_idx])
                    positives["true_label"].append(ground_truth[i]["labels"][true_idx])
                    positives["pred_label"].append(predictions[i]["labels"][pred_idx])
                    positives["correct"].append(int(correct))
                    positives["image_id"].append(ground_truth[i]["image_id"][0])

                negatives[true_idx] = max(negatives[true_idx],
                                        predictions[i]["scores"][pred_idx]*correct)

            if not used: 
                positives["iou"].append(0)
                positives["score"].append(predictions[i]["scores"][pred_idx])
                positives["true_label"].append(-1)
                positives["pred_label"].append(predictions[i]["labels"][pred_idx])
                positives["image_id"].append(ground_truth[i]["image_id"][0])
                positives["correct"].append(0)

    for key in positives:
        positives[key] = np.array(positives[key])
    
    negatives = np.array([val for val in negatives.values()])
    
    return positives, negatives

def pr_curve(positives, negatives):
    thresholds = np.linspace(0, 1, 11)
    metrics = []
    for score_threshold in thresholds:
        mask = positives["score"] > score_threshold
        if mask.sum() > 0:
            precision = np.sum(positives["correct"][mask]) / np.sum(mask)
        else:
            precision = 0

        mask = negatives > score_threshold
        recall = np.sum(mask) / negatives.shape[0]
        
        metrics.append([precision, recall])
    return metrics


@torch.no_grad()
def evaluate(model, loader, device, epoch, iou_threshold=0.5, target_recall=0.95):
    model.eval()

    predictions = []
    ground_truth = []
    for images, targets in tqdm(loader, total=len(loader)):
        images = [image.to(device) for image in images]
        #print("run model...")
        out = model(images, targets)
        
        predictions.extend([{k:v.cpu() for k, v in o.items()} for o in out])
        ground_truth.extend(targets)

    positives, negatives = positives_and_negatives(
                                    ground_truth, predictions, iou_threshold)
    metrics = pr_curve(positives, negatives)
    
    precision_at_target_recall = 0
    average_precision = 0
    prev_p, prev_r = 0, 1
    for p, r in metrics:
        average_precision += (p - prev_p) * (prev_r - r)
        prev_p, prev_r = p, r
        if r > target_recall:
            precision_at_target_recall = p
    average_precision /= len(metrics)

    pr_table = wandb.Table(data=metrics, columns = ["precision", "recall"])
    log_dict = {
        f"AP@{iou_threshold}": average_precision,
        f"PR-curve": wandb.plot.line(pr_table, "recall", "precision", title="PR-curve"),
        f"precision@{target_recall}": precision_at_target_recall,
        f"epoch": epoch
    }


    def wandbBBoxes(bboxes, scores, labels):
        all_bboxes = []
        for box, score, label in zip(bboxes, scores, labels):
            all_bboxes.append({
                "position" : {
                    "minX" : int(box[0]),
                    "minY" : int(box[1]),
                    "maxX" : int(box[2]),
                    "maxY" : int(box[3])
                },
                "class_id" : int(label.item()),
                "box_caption" : f"{loader.dataset.to_name[label]} ({score:.3f})",
                "scores" : { "score" : score.item() },
                "domain" : "pixel",
            })
        return all_bboxes

    for i, (image, target, pred) in enumerate(zip(images, targets, out)):
        log_dict.update({
            f"examples/example-{i:02d}" : wandb.Image(
                image.cpu().numpy().transpose([1, 2, 0]),
                boxes = {
                    "predictions": {
                        "box_data": wandbBBoxes(
                            pred["boxes"].cpu().numpy(),
                            pred["scores"].cpu().numpy(),
                            pred["labels"].cpu().numpy()
                        ),
                        "class_labels": loader.dataset.to_name
                    },
                    "ground_truth": {
                        "box_data": wandbBBoxes(
                            target["boxes"].numpy(),
                            np.ones_like(target["labels"].numpy()),
                            target["labels"].numpy()
                        ),
                        "class_labels": loader.dataset.to_name
                    }
                }
            )
        })

    wandb.log(log_dict)
    return log_dict