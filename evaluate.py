import torch
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.ops import box_iou
import torchvision
from PIL import Image

# --- Imports from your project ---
from backbone.radar.radar_encoder import RCNetWithTransformer, RCNet 
from detection.detection_head import NanoDetectionHead
from model import RadarDetectionModel
from data.WaterScenesDataset import WaterScenesDataset, collate_fn

# ==========================================
#               CONFIGURATION
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATASET_ROOT = os.path.abspath("./data/WaterScenes")
TEST_FILE = os.path.join(DATASET_ROOT, "test.txt")

# --- EXPERIMENT CONFIGURATION ---
EXPERIMENTS = [
    {
        "name": "Half Transformer Model",
        "path": "./checkpoints/rcnet_radar_detection_half_transformer.pth", 
        "config": {
            "backbone_type": "RCNetWithTransformer",
            "phi": "S0",
            "last_stages_only": True 
        }
    },
    {
        "name": "Half Transformer Model Transfer Learning", 
        "path": "./checkpoints/rcnet_radar_detection_half_transformer_transfer_learning.pth", 
        "config": {
            "backbone_type": "RCNetWithTransformer",
            "phi": "S0",
            "last_stages_only": True 
        }
    },
    {
        "name": "Full Transformer Model", 
        "path": "./checkpoints/rcnet_radar_detection_full_transformer.pth",
        "config": {
            "backbone_type": "RCNetWithTransformer",
            "phi": "S0",
            "last_stages_only": False
        }
    }
]

# Common Config
NUM_CLASSES = 7
TARGET_SIZE = (320, 320)
RADAR_MEAN = [0.1127, -0.0019, -0.0012, 0.0272]
RADAR_STD  = [3.1396,  0.2177,  0.0556,  0.6252]
CONF_THRESH = 0.05
NMS_THRESH  = 0.45
IOU_THRESH  = 0.25
CLASS_NAMES = ["Pier", "Buoy", "Sailor", "Ship", "Boat", "Vessel", "Kayak"]

# ==========================================
#           HELPER FUNCTIONS
# ==========================================

def build_model_from_config(config):
    """Factory function to build the correct model architecture"""
    in_channels = 4
    phi = config.get('phi', 'S0')
    backbone_type = config.get('backbone_type', 'RCNet')
    
    # Get the flag (Default to True if not specified, assuming newer models)
    last_stages_only = config.get('last_stages_only', True)
    
    if backbone_type == 'RCNet':
        backbone = RCNet(in_channels=in_channels, phi=phi)
    elif backbone_type == 'RCNetWithTransformer':
        print(f"   -> Building RCNetWithTransformer (last_stages_only={last_stages_only})")
        backbone = RCNetWithTransformer(
            in_channels=in_channels, 
            phi=phi, 
            max_input_hw=320,
            last_stages_only=last_stages_only # <--- Pass the flag here
        )
    else:
        raise ValueError(f"Unknown backbone type: {backbone_type}")
    
    # S0 Channels
    head_channels = [12, 24, 44] 

    head = NanoDetectionHead(num_classes=NUM_CLASSES, in_channels_list=head_channels, head_width=32)
    model = RadarDetectionModel(backbone, head)
    return model

# ... (The rest of the functions: box_iou, non_max_suppression, decode_outputs, etc. remain EXACTLY the same) ...
# Copy them from the previous script or ask me if you need them pasted again.

def decode_outputs(outputs, strides=[8, 16, 32]):
    decoded = []
    for i, output in enumerate(outputs):
        B, C, H, W = output.shape
        stride = strides[i]
        yv, xv = torch.meshgrid([torch.arange(H), torch.arange(W)], indexing='ij')
        grid = torch.stack((xv, yv), 2).view(1, -1, 2).to(output.device).float()
        output = output.flatten(start_dim=2).permute(0, 2, 1)
        output[..., :2] = (output[..., :2] + grid) * stride
        output[..., 2:4] = torch.exp(output[..., 2:4]) * stride
        decoded.append(output)
    return torch.cat(decoded, dim=1)

def non_max_suppression(prediction, conf_thres=0.25, nms_thres=0.45):
    output = [None] * len(prediction)
    for image_i, pred in enumerate(prediction):
        pred[:, 4] = torch.sigmoid(pred[:, 4])
        pred[:, 5:] = torch.sigmoid(pred[:, 5:])
        pred_score, pred_cls = torch.max(pred[:, 5:], 1, keepdim=True)
        combined_score = pred[:, 4:5] * pred_score
        mask = (combined_score > conf_thres).squeeze()
        pred = pred[mask]
        combined_score = combined_score[mask]
        pred_cls = pred_cls[mask]
        if len(pred) == 0: continue
        box = pred[:, :4].clone()
        box[:, 0] = pred[:, 0] - pred[:, 2] / 2
        box[:, 1] = pred[:, 1] - pred[:, 3] / 2
        box[:, 2] = pred[:, 0] + pred[:, 2] / 2
        box[:, 3] = pred[:, 1] + pred[:, 3] / 2
        detections = torch.cat((box, combined_score, pred_cls.float()), 1)
        keep = torchvision.ops.nms(detections[:, :4], detections[:, 4], nms_thres)
        output[image_i] = detections[keep]
    return output

def compute_ap(recall, precision):
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def get_batch_statistics(outputs, targets, iou_threshold):
    batch_metrics = []
    for sample_i in range(len(outputs)):
        if outputs[sample_i] is None:
            continue

        output = outputs[sample_i] # [x1, y1, x2, y2, score, class]
        pred_boxes = output[:, :4]
        pred_scores = output[:, 4]
        pred_labels = output[:, 5]

        true_positives = torch.zeros(pred_boxes.shape[0]) # Created on CPU

        # Get annotations
        annotations = targets[targets[:, 0] == sample_i][:, 1:]
        target_labels = annotations[:, 0] if len(annotations) else []
        
        if len(annotations):
            detected_boxes = []
            target_boxes = annotations[:, 1:]
            
            t_boxes = target_boxes.clone()
            # Convert target (cx, cy, w, h) -> (x1, y1, x2, y2)
            t_boxes[:, 0] = target_boxes[:, 0] - target_boxes[:, 2] / 2
            t_boxes[:, 1] = target_boxes[:, 1] - target_boxes[:, 3] / 2
            t_boxes[:, 2] = target_boxes[:, 0] + target_boxes[:, 2] / 2
            t_boxes[:, 3] = target_boxes[:, 1] + target_boxes[:, 3] / 2

            for pred_i, (pred_box, pred_label) in enumerate(zip(pred_boxes, pred_labels)):
                
                if len(detected_boxes) == len(annotations):
                    break

                if pred_label not in target_labels:
                    continue

                # Use torchvision.ops.box_iou for stability
                # Ensure dimensions match: [1, 4] vs [M, 4]
                iou = torchvision.ops.box_iou(pred_box.unsqueeze(0), t_boxes).squeeze(0)
                max_iou, box_index = iou.max(0)
                
                if max_iou >= iou_threshold and box_index not in detected_boxes:
                    true_positives[pred_i] = 1
                    detected_boxes.append(box_index)
        
        # --- FIX: Move everything to CPU to match true_positives ---
        batch_metrics.append([true_positives, pred_scores.cpu(), pred_labels.cpu()])
        
    return batch_metrics

def evaluate_experiment(experiment, test_loader):
    name = experiment['name']
    path = experiment['path']
    config = experiment['config']
    
    print(f"\n--- Evaluating: {name} ---")
    
    try:
        model = build_model_from_config(config).to(DEVICE)
    except Exception as e:
        print(f"Error building model: {e}")
        return None, None

    try:
        model.load_state_dict(torch.load(path, map_location=DEVICE))
    except Exception as e:
        print(f"Error loading weights: {e}")
        return None, None
    
    model.eval()
    
    labels = []
    sample_metrics = [] 

    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"Eval {name}"):
            radars = batch['radar'].to(DEVICE)
            targets = batch['label'].to(DEVICE)
            
            preds_raw = model(radars)
            preds_decoded = decode_outputs(preds_raw)
            preds_nms = non_max_suppression(preds_decoded, CONF_THRESH, NMS_THRESH)
            
            sample_metrics += get_batch_statistics(preds_nms, targets, IOU_THRESH)
            
            for i in range(len(preds_nms)):
                annots = targets[targets[:, 0] == i]
                labels += annots[:, 1].tolist()

    if len(sample_metrics) == 0:
        return 0, [0]*NUM_CLASSES

    true_positives, pred_scores, pred_labels = [torch.cat(x, 0) for x in list(zip(*sample_metrics))]
    precision, recall, AP, f1, ap_class = [], [], [], [], []
    
    for c in range(NUM_CLASSES):
        class_mask = pred_labels == c
        n_gt = labels.count(c)
        if n_gt == 0 and class_mask.sum() == 0: continue 
        if n_gt == 0 and class_mask.sum() > 0:
            AP.append(0)
            continue

        tp = true_positives[class_mask]
        conf = pred_scores[class_mask]
        conf, sort_ind = torch.sort(conf, descending=True)
        tp = tp[sort_ind]
        fp = 1 - tp
        tp = torch.cumsum(tp, dim=0)
        fp = torch.cumsum(fp, dim=0)
        rec = tp / (n_gt + 1e-16)
        prec = tp / (tp + fp + 1e-16)
        ap = compute_ap(rec.cpu().numpy(), prec.cpu().numpy())
        AP.append(ap)
        ap_class.append(c)

    mAP = np.mean(AP) if len(AP) > 0 else 0
    print(f"    -> mAP@0.5: {mAP:.4f}")
    return mAP, AP

def save_qualitative_results(experiment, dataset, output_dir="report_figures"):
    name = experiment['name']
    path = experiment['path']
    config = experiment['config']
    
    model = build_model_from_config(config).to(DEVICE)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.eval()
    
    indices = np.random.choice(len(dataset), 3, replace=False)
    
    for idx in indices:
        sample = dataset[idx]
        file_id = sample['file_id']
        radar = sample['radar'].unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            raw = model(radar)
            decoded = decode_outputs(raw)
            results = non_max_suppression(decoded, conf_thres=0.3, nms_thres=0.45)[0] 

        img_path = os.path.join(DATASET_ROOT, 'image', f"{file_id}.jpg")
        try:
            pil_img = Image.open(img_path).convert("RGB").resize((320, 320))
            img_draw = np.array(pil_img)
        except:
            img_draw = np.zeros((320, 320, 3), dtype=np.uint8)

        if results is not None:
            for box in results:
                x1, y1, x2, y2, score, cls_id = box.cpu().numpy()
                cv2.rectangle(img_draw, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                label = f"{CLASS_NAMES[int(cls_id)]} {score:.2f}"
                cv2.putText(img_draw, label, (int(x1), int(y1)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(sample['radar'][3].cpu(), cmap='magma') 
        ax[0].set_title("Radar Power")
        ax[0].axis('off')
        ax[1].imshow(img_draw)
        ax[1].set_title(f"{name}: {file_id}")
        ax[1].axis('off')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/viz_{name}_{file_id}.png")
        plt.close()

if __name__ == "__main__":
    
    img_tf = transforms.Compose([
        transforms.Resize(TARGET_SIZE), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_dataset = WaterScenesDataset(DATASET_ROOT, TEST_FILE, image_transform=img_tf, radar_mean=RADAR_MEAN, radar_std=RADAR_STD)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4, collate_fn=collate_fn)

    results = {}
    os.makedirs("report_figures", exist_ok=True)

    for exp in EXPERIMENTS:
        path = exp['path']
        if os.path.exists(path):
            mAP, AP = evaluate_experiment(exp, test_loader)
            if mAP is not None:
                results[exp['name']] = {'mAP': mAP, 'AP': AP}
                save_qualitative_results(exp, test_dataset)
        else:
            print(f"Skipping {exp['name']}: Path {path} not found.")

    if results:
        names = list(results.keys())
        maps = [results[n]['mAP'] for n in names]
        plt.figure(figsize=(10, 6))
        plt.bar(names, maps, color='skyblue')
        plt.ylabel('mAP @ 0.5')
        plt.title('Performance Comparison')
        plt.savefig('report_figures/comparison_bar.png')
        print("Comparison chart saved.")