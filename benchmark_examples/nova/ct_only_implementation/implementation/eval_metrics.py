import re
import nltk
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import evaluate
import numpy as np
import json

def extract_json(text):
    """Extracts JSON string from text, handling markdown code blocks."""
    match = re.search(r'```json\s*(.*)', text, re.DOTALL)
    if match:
        content = match.group(1)
        # remove closing ``` if present
        content = re.sub(r'\s*```$', '', content)
        return content.strip()
    return text.strip()



def parse_boxes(box_string):
    """Parses a bounding box string like '[ymax, xmin, xmax, ymin]' and returns COCO format."""
    try:
        if box_string.strip() == "no target":
            return []
        
        lines = box_string.strip().split("\n")
        boxes = []
        for line in lines:
            if ":" in line and "[" in line and "]" in line:
                try:
                    coords_str = line.split(":", 1)[1].strip().strip("[]")
                    c = [float(n.strip()) for n in coords_str.split(",")]
                    if len(c) == 4:
                        # Robust mapping using min/max
                        ymin = min(c[0], c[2])
                        ymax = max(c[0], c[2])
                        xmin = min(c[1], c[3])
                        xmax = max(c[1], c[3])
                        
                        # Rescale to 480x480
                        xmin = xmin * 0.48
                        ymin = ymin * 0.48
                        xmax = xmax * 0.48
                        ymax = ymax * 0.48
                        
                        width = xmax - xmin
                        height = ymax - ymin
                        
                        boxes.append([xmin, ymin, width, height])
                except Exception as e:
                    print(f"Failed to parse line: {line}. Error: {e}")
        return boxes
    except Exception as e:
        print(f"Error parsing boxes: {e}")
        raise e





def calculate_iou(boxA, boxB):
    """Calculates IoU of two boxes given as [xmin, ymin, width, height]."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    interWidth = max(0, xB - xA)
    interHeight = max(0, yB - yA)
    interArea = interWidth * interHeight

    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]

    unionArea = float(boxAArea + boxBArea - interArea)
    if unionArea == 0:
        return 0.0
    return interArea / unionArea


def calculate_coco_metrics(all_gt_boxes, all_pred_boxes):
    """Calculates COCO metrics using pycocotools."""
    coco_dict = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "abnormality"}]
    }
    
    predictions = []
    
    ann_id = 1
    for img_id, (gt_boxes, pred_boxes) in enumerate(zip(all_gt_boxes, all_pred_boxes), start=1):
        coco_dict["images"].append({"id": img_id, "width": 480, "height": 480})
        
        for gb in gt_boxes:
            coco_dict["annotations"].append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": 1,
                "bbox": gb,
                "area": gb[2] * gb[3],
                "iscrowd": 0
            })
            ann_id += 1
            
        for pb in pred_boxes:
            predictions.append({
                "image_id": img_id,
                "category_id": 1,
                "bbox": pb,
                "score": 1.0  # Dummy score
            })
            
    coco_gt = COCO()
    coco_gt.dataset = coco_dict
    coco_gt.createIndex()
    
    if predictions:
        coco_dt = coco_gt.loadRes(predictions)
        
        # Evaluate at 0.3 IoU
        coco_eval_30 = COCOeval(coco_gt, coco_dt, 'bbox')
        coco_eval_30.params.iouThrs = np.array([0.3])
        coco_eval_30.evaluate()
        coco_eval_30.accumulate()
        coco_eval_30.summarize()
        
        # Evaluate at 0.5:0.95 IoU
        coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        
        return {
            "mAP@30": float(coco_eval_30.stats[0]),
            "mAP@50": float(coco_eval.stats[1]),
            "mAP@[50:95]": float(coco_eval.stats[0])
        }
    else:
        return {
            "mAP@30": 0.0,
            "mAP@50": 0.0,
            "mAP@[50:95]": 0.0
        }




def calculate_counts(gt_boxes, pred_boxes, iou_threshold=0.30):
    """Computes TP, FP, FN."""
    if not gt_boxes:
        return 0, len(pred_boxes), 0

    tp = 0
    matched_gt = set()

    for pb in pred_boxes:
        best_iou = 0
        best_gt_idx = -1
        for g_idx, gb in enumerate(gt_boxes):
            if g_idx in matched_gt:
                continue
            iou = calculate_iou(pb, gb)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = g_idx

        if best_iou >= iou_threshold:
            tp += 1
            matched_gt.add(best_gt_idx)

    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - len(matched_gt)

    return tp, fp, fn



def calculate_caption_metrics(text, reference):
    """Calculates BLEU-4 and METEOR scores using HuggingFace evaluate."""
    # Use HuggingFace evaluate
    bleu_metric = evaluate.load("bleu")
    meteor_metric = evaluate.load("meteor")

    try:
        bleu_res = bleu_metric.compute(predictions=[text], references=[[reference]])
        bleu = bleu_res.get("bleu", 0.0)
    except Exception:
        bleu = 0.0

    try:
        meteor_res = meteor_metric.compute(predictions=[text], references=[reference])
        met = meteor_res.get("meteor", 0.0)
    except Exception:
        met = 0.0

    return bleu, met






