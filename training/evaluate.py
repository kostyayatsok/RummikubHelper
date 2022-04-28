import numpy as np
import wandb
import torch
from tqdm import tqdm
from torchvision.ops.boxes import box_iou
from collections import defaultdict

def positives_and_negatives(ground_truth, predictions, iou_threshold=0.5):
    positives = defaultdict(list)
    negatives = []

    for i in range(len(predictions)):
        negatives.append({})
        for true_idx in range(len(ground_truth[i]["boxes"])):            
            negatives[i][true_idx] = -1
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

                    if correct:
                        negatives[i][true_idx] = max(
                            negatives[i][true_idx],
                            predictions[i]["scores"][pred_idx]
                        )

            if not used: 
                positives["iou"].append(0)
                positives["score"].append(predictions[i]["scores"][pred_idx])
                positives["true_label"].append(-1)
                positives["pred_label"].append(predictions[i]["labels"][pred_idx])
                positives["image_id"].append(ground_truth[i]["image_id"][0])
                positives["correct"].append(0)

    for key in positives:
        positives[key] = np.array(positives[key])
    
    negatives = np.array([[val for val in img.values()] for img in negatives])
    negatives = np.concatenate(negatives)
    return positives, negatives

def pr_curve(positives, negatives):
    thresholds = np.linspace(0, 1, 11)
    metrics = []
    for score_threshold in thresholds:
        mask = positives["score"] >= score_threshold
        if mask.sum() > 0:
            precision = np.sum(positives["correct"][mask]) / np.sum(mask)
        else:
            precision = 1

        mask = negatives >= score_threshold
        recall = np.sum(mask) / negatives.shape[0]
        
        metrics.append([precision, recall])
    return metrics

def wandbBBoxes(bboxes, scores, labels, to_name):
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
            "box_caption" : f"{to_name[label]} ({score:.3f})",
            "scores" : { "score" : score.item() },
            "domain" : "pixel",
        })
    return all_bboxes


@torch.no_grad()
def evaluate(loader, device, epoch, model=None, model_out=None, iou_threshold=0.5, target_recall=0.95):
    if model is not None:
        model.eval()
    elif model_out is not None:
        model_out["bbox"] = model_out["bbox"].apply(lambda x: [x[0], x[1], x[0]+x[2], x[1]+x[3]])
        model_out = model_out.groupby("image_id").agg(list)
        model_out = model_out.rename(columns={
            "bbox": "boxes",
            "score": "scores",
            "category_id": "labels"
        })
    else:
        raise "You sholud provide either model or model_out"
    predictions = []
    ground_truth = []
    for images, targets in tqdm(loader, total=len(loader)):
        images = [image.to(device) for image in images]

        if model is not None:
            out = model(images, targets)
            out = [{k:v.cpu() for k, v in o.items()} for o in out]
        else:
            out = []
            for gt in targets:
                image_id = gt["image_id"][0].item()
                if image_id in model_out.index:
                    out.append(model_out.loc[image_id].to_dict())
                else:
                    out.append({"boxes":[], "scores":[], "labels":[]})

            out = [{k:torch.tensor(v) for k, v in o.items()} for o in out]

            
        for image, target, pred in zip(images, targets, out):
            wandb.log({
                f"examples/example" : wandb.Image(
                    image.cpu().numpy().transpose([1, 2, 0]),
                    boxes = {
                        "predictions": {
                            "box_data": wandbBBoxes(
                                pred["boxes"].cpu().numpy(),
                                pred["scores"].cpu().numpy(),
                                pred["labels"].cpu().numpy(),
                                loader.dataset.to_name
                            ),
                            "class_labels": loader.dataset.to_name
                        },
                        "ground_truth": {
                            "box_data": wandbBBoxes(
                                target["boxes"].numpy(),
                                np.ones_like(target["labels"].numpy()),
                                target["labels"].numpy(),
                                loader.dataset.to_name
                            ),
                            "class_labels": loader.dataset.to_name
                        }
                    }
                )
            })


        predictions.extend(out)
        ground_truth.extend(targets)

    positives, negatives = positives_and_negatives(
                                    ground_truth, predictions, iou_threshold)
    metrics = pr_curve(positives, negatives)
    
    precision_at_target_recall = 0
    average_precision = 0
    prev_p = 0
    for p, r in metrics:
        average_precision += (p - prev_p) * r
        prev_p = p
        if r > target_recall:
            precision_at_target_recall = p

    print(average_precision)
    pr_table = wandb.Table(data=metrics, columns = ["precision", "recall"])
    log_dict = {
        f"AP@{iou_threshold}": average_precision,
        f"PR-curve": wandb.plot.line(pr_table, "recall", "precision", title="PR-curve"),
        f"precision@{target_recall}": precision_at_target_recall,
        f"epoch": epoch
    }

    # wandb.log(log_dict)
    return log_dict