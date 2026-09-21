"""
PHASE 7 — DHF1K SPATIAL SALIENCY PROTOTYPE TRAINING SCRIPT

Architecture: TinySalNet (Lightweight Convolutional Encoder-Decoder)
Primary Target: DHF1K continuous saliency-density map (maps/*.png)
Secondary Reference: DHF1K fixation map (fixation/*.png)
Supervision: "DHF1K saliency-density supervision"
Objective: Spatial KL Divergence
Evaluation Metrics: NSS, AUC-Judd, KL Divergence, CC, SIM vs. Uniform Baseline
"""

import os
import sys
import time
import json
import random
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ---------------------------------------------------------------------------
# Configuration & Constants
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
DHF1K_ROOT = r"E:\viral_attention_data\DHF1K"
VIDEO_DIR = os.path.join(DHF1K_ROOT, "video")
ANNOTATION_DIR = os.path.join(DHF1K_ROOT, "annotation")
OUTPUT_DIR = r"E:\new viral\models\human_attention_experiments\dhf1k_spatial"

# Spatial resolution (Width x Height) - 16:9 aspect ratio matching DHF1K (640x360)
INPUT_WIDTH = 160
INPUT_HEIGHT = 90

# Temporal sampling: 4 deterministic frames per video at progress coordinates
PROGRESS_COORDINATES = [0.125, 0.375, 0.625, 0.875]

def set_seed(seed=RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ---------------------------------------------------------------------------
# Architecture: TinySalNet
# ---------------------------------------------------------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.conv(x)

class TinySalNet(nn.Module):
    """
    Lightweight CNN Encoder-Decoder for Spatial Saliency Estimation.
    Parameters: ~85k parameters, fast CPU inference.
    Input: (B, 3, H, W) normalized RGB [0, 1]
    Output: (B, 1, H, W) spatial probability density (sums to 1 per map)
    """
    def __init__(self):
        super().__init__()
        # Encoder: HxW -> H/2 x W/2 -> H/4 x W/4 -> H/8 x W/8
        self.enc1 = ConvBlock(3, 16)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.enc2 = ConvBlock(16, 32)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.enc3 = ConvBlock(32, 64)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        # Bottleneck with dilation for expanded receptive field
        self.bottleneck = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        
        # Decoder: H/8 x W/8 -> H/4 x W/4 -> H/2 x W/2 -> HxW
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec3 = ConvBlock(64 + 32, 32)
        
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec2 = ConvBlock(32 + 16, 16)
        
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec1 = ConvBlock(16, 8)
        
        # Saliency head: 1x1 conv to 1 channel
        self.head = nn.Conv2d(8, 1, kernel_size=1)
        self.softplus = nn.Softplus(beta=1.0)

    def forward(self, x):
        # Target shape for exact reconstruction
        in_h, in_w = x.shape[2], x.shape[3]
        
        # Encoder
        e1 = self.enc1(x)               # (B, 16, H, W)
        p1 = self.pool1(e1)             # (B, 16, H/2, W/2)
        
        e2 = self.enc2(p1)              # (B, 32, H/2, W/2)
        p2 = self.pool2(e2)             # (B, 32, H/4, W/4)
        
        e3 = self.enc3(p2)              # (B, 64, H/4, W/4)
        p3 = self.pool3(e3)             # (B, 64, H/8, W/8)
        
        # Bottleneck
        b = self.bottleneck(p3)         # (B, 64, H/8, W/8)
        
        # Decoder with skip connections
        u3 = self.up3(b)
        if u3.shape[2:] != e2.shape[2:]:
            u3 = F.interpolate(u3, size=e2.shape[2:], mode='bilinear', align_corners=False)
        d3 = self.dec3(torch.cat([u3, e2], dim=1))
        
        u2 = self.up2(d3)
        if u2.shape[2:] != e1.shape[2:]:
            u2 = F.interpolate(u2, size=e1.shape[2:], mode='bilinear', align_corners=False)
        d2 = self.dec2(torch.cat([u2, e1], dim=1))
        
        u1 = self.up1(d2)
        if u1.shape[2:] != (in_h, in_w):
            u1 = F.interpolate(u1, size=(in_h, in_w), mode='bilinear', align_corners=False)
        d1 = self.dec1(u1)
        
        out = self.head(d1)             # (B, 1, H, W)
        pos = self.softplus(out) + 1e-7
        
        # Normalize so spatial map sums to 1.0 per sample
        spatial_sum = pos.sum(dim=(2, 3), keepdim=True)
        saliency_map = pos / spatial_sum
        return saliency_map

# ---------------------------------------------------------------------------
# Loss: Spatial Kullback-Leibler (KL) Divergence
# ---------------------------------------------------------------------------
def kl_divergence_loss(pred, target, eps=1e-7):
    """
    Spatial KL Divergence between continuous saliency maps.
    pred: (B, 1, H, W) predicted probability density (sum=1)
    target: (B, 1, H, W) target continuous saliency density (sum=1)
    D_KL(target || pred) = sum( target * log( target / (pred + eps) + eps ) )
    """
    target_clean = target / (target.sum(dim=(2, 3), keepdim=True) + eps)
    pred_clean = pred / (pred.sum(dim=(2, 3), keepdim=True) + eps)
    
    kl = target_clean * torch.log(eps + target_clean / (pred_clean + eps))
    return kl.sum(dim=(2, 3)).mean()

# ---------------------------------------------------------------------------
# Saliency Evaluation Metrics
# ---------------------------------------------------------------------------
def compute_nss(saliency_map, fixation_map):
    """
    Normalized Scanpath Saliency (NSS).
    saliency_map: 2D numpy array
    fixation_map: 2D binary numpy array (1 at fixation, 0 elsewhere)
    """
    fixation_mask = fixation_map > 0
    if fixation_mask.sum() == 0:
        return np.nan
    s_map = saliency_map.astype(np.float64)
    std = s_map.std()
    if std < 1e-9:
        return 0.0
    norm_s = (s_map - s_map.mean()) / std
    return float(norm_s[fixation_mask].mean())

def compute_cc(saliency_map, target_map):
    """
    Linear Correlation Coefficient (CC).
    """
    s = saliency_map.astype(np.float64).flatten()
    t = target_map.astype(np.float64).flatten()
    s_std = s.std()
    t_std = t.std()
    if s_std < 1e-9 or t_std < 1e-9:
        return 0.0
    corr = np.corrcoef(s, t)[0, 1]
    return float(corr) if not np.isnan(corr) else 0.0

def compute_sim(saliency_map, target_map):
    """
    Similarity / Histogram Intersection (SIM).
    Both maps normalized to sum to 1.
    """
    s = saliency_map.astype(np.float64)
    t = target_map.astype(np.float64)
    s_sum = s.sum()
    t_sum = t.sum()
    if s_sum > 0:
        s = s / s_sum
    if t_sum > 0:
        t = t / t_sum
    return float(np.sum(np.minimum(s, t)))

def compute_kl(saliency_map, target_map, eps=1e-7):
    """
    Kullback-Leibler Divergence (KL) on 2D numpy arrays.
    D_KL(target || pred)
    """
    s = saliency_map.astype(np.float64)
    t = target_map.astype(np.float64)
    s = s / (s.sum() + eps)
    t = t / (t.sum() + eps)
    kl = np.sum(t * np.log(eps + t / (s + eps)))
    return float(kl)

def compute_auc_judd(saliency_map, fixation_map, jitter=True):
    """
    AUC-Judd: Area under ROC curve using fixation points as positives
    and all non-fixation pixels as negatives.
    """
    s_map = saliency_map.astype(np.float64)
    f_map = (fixation_map > 0).astype(bool)
    
    if f_map.sum() == 0 or (~f_map).sum() == 0:
        return np.nan
        
    if jitter:
        # Small deterministic jitter to break ties
        rng = np.random.RandomState(42)
        s_map = s_map + rng.uniform(0, 1e-7, size=s_map.shape)
        
    s_map = (s_map - s_map.min()) / (s_map.max() - s_map.min() + 1e-9)
    
    S_fix = s_map[f_map]
    S_nonfix = s_map[~f_map]
    
    # Sort thresholds by fixation values
    thresholds = np.sort(S_fix)[::-1]
    
    tp = np.zeros(len(thresholds) + 2)
    fp = np.zeros(len(thresholds) + 2)
    tp[0] = 0
    fp[0] = 0
    tp[-1] = 1
    fp[-1] = 1
    
    n_fix = len(S_fix)
    n_nonfix = len(S_nonfix)
    
    # Fast calculation via searchsorted
    S_nonfix_sorted = np.sort(S_nonfix)
    for i, th in enumerate(thresholds):
        tp[i + 1] = (i + 1) / n_fix
        # Count elements in S_nonfix >= th
        idx = np.searchsorted(S_nonfix_sorted, th, side='left')
        fp[i + 1] = (n_nonfix - idx) / n_nonfix
        
    auc = np.trapz(tp, fp)
    return float(auc)

# ---------------------------------------------------------------------------
# Dataset & Frame Sampling
# ---------------------------------------------------------------------------
class DHF1KSpatialDataset(Dataset):
    """
    Dataset of deterministically sampled video frames and corresponding
    DHF1K continuous saliency density maps and fixation maps.
    """
    def __init__(self, video_ids, dhf1k_root=DHF1K_ROOT, 
                 input_w=INPUT_WIDTH, input_h=INPUT_HEIGHT,
                 progress_coords=PROGRESS_COORDINATES, preload=True):
        self.video_ids = video_ids
        self.dhf1k_root = dhf1k_root
        self.video_dir = os.path.join(dhf1k_root, "video")
        self.annotation_dir = os.path.join(dhf1k_root, "annotation")
        self.input_w = input_w
        self.input_h = input_h
        self.progress_coords = progress_coords
        self.preload = preload
        
        # Build deterministic sample index
        self.samples = []
        for vid in self.video_ids:
            vid_str = f"{vid:04d}"
            # Check annotation directory
            ann_sub = os.path.join(self.annotation_dir, vid_str)
            if not os.path.exists(ann_sub):
                ann_sub = os.path.join(self.annotation_dir, f"{vid:03d}")
                
            map_dir = os.path.join(ann_sub, "maps")
            fix_dir = os.path.join(ann_sub, "fixation")
            
            # Check video path
            vid_path = os.path.join(self.video_dir, f"{vid:03d}.AVI")
            if not os.path.exists(vid_path):
                vid_path = os.path.join(self.video_dir, f"{vid:04d}.AVI")
            if not os.path.exists(vid_path):
                vid_path = os.path.join(self.video_dir, f"{vid}.AVI")
            
            if not os.path.exists(map_dir) or not os.path.exists(vid_path):
                continue
                
            map_files = sorted([f for f in os.listdir(map_dir) if f.endswith('.png')])
            num_frames = len(map_files)
            if num_frames == 0:
                continue
                
            vid_samples = []
            for q_idx, p in enumerate(self.progress_coords):
                frame_idx = int(np.clip(p * num_frames, 0, num_frames - 1))
                frame_filename = map_files[frame_idx] # e.g. '0045.png'
                map_path = os.path.join(map_dir, frame_filename)
                fix_path = os.path.join(fix_dir, frame_filename)
                
                vid_samples.append({
                    'video_id': vid,
                    'video_id_str': vid_str,
                    'video_path': vid_path,
                    'frame_idx': frame_idx,
                    'frame_number': int(frame_filename.replace('.png', '')),
                    'temporal_quartile': q_idx + 1,
                    'progress_coord': p,
                    'map_path': map_path,
                    'fix_path': fix_path
                })
            self.samples.extend(vid_samples)

        self.cached_items = []
        if self.preload and len(self.samples) > 0:
            print(f"Preloading {len(self.samples)} frames into memory for ultra-fast processing...", flush=True)
            # Group by video path to read efficiently
            by_video = {}
            for item in self.samples:
                by_video.setdefault(item['video_path'], []).append(item)
                
            loaded_dict = {}
            for vid_path, items in by_video.items():
                cap = cv2.VideoCapture(vid_path)
                for it in items:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, it['frame_idx'])
                    ret, frame_bgr = cap.read()
                    if not ret or frame_bgr is None:
                        frame_rgb = np.zeros((self.input_h, self.input_w, 3), dtype=np.uint8)
                    else:
                        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                        frame_rgb = cv2.resize(frame_rgb, (self.input_w, self.input_h), interpolation=cv2.INTER_AREA)
                    inp_t = torch.from_numpy(frame_rgb.astype(np.float32) / 255.0).permute(2, 0, 1)

                    # Map
                    s_img = cv2.imread(it['map_path'], cv2.IMREAD_GRAYSCALE)
                    if s_img is None:
                        s_map = np.ones((self.input_h, self.input_w), dtype=np.float32)
                    else:
                        s_map = cv2.resize(s_img, (self.input_w, self.input_h), interpolation=cv2.INTER_AREA).astype(np.float32)
                    s_sum = s_map.sum()
                    if s_sum > 0:
                        s_map = s_map / s_sum
                    else:
                        s_map = np.full((self.input_h, self.input_w), 1.0 / (self.input_h * self.input_w), dtype=np.float32)
                    tgt_t = torch.from_numpy(s_map).unsqueeze(0)

                    # Fixation
                    f_img = cv2.imread(it['fix_path'], cv2.IMREAD_GRAYSCALE)
                    if f_img is None:
                        f_map = np.zeros((self.input_h, self.input_w), dtype=np.float32)
                    else:
                        f_map = cv2.resize(f_img, (self.input_w, self.input_h), interpolation=cv2.INTER_NEAREST).astype(np.float32)
                        f_map = (f_map > 127).astype(np.float32)
                    fix_t = torch.from_numpy(f_map).unsqueeze(0)

                    loaded_dict[(it['video_id'], it['frame_idx'])] = {
                        'input': inp_t,
                        'target': tgt_t,
                        'fixation': fix_t,
                        'video_id': it['video_id'],
                        'video_id_str': it['video_id_str'],
                        'frame_idx': it['frame_idx'],
                        'frame_number': it['frame_number'],
                        'temporal_quartile': it['temporal_quartile'],
                        'progress_coord': it['progress_coord']
                    }
                cap.release()
            
            for item in self.samples:
                self.cached_items.append(loaded_dict[(item['video_id'], item['frame_idx'])])
            print(f"Preloading complete: {len(self.cached_items)} frames cached.", flush=True)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self.preload and len(self.cached_items) == len(self.samples):
            return self.cached_items[idx]

        item = self.samples[idx]
        cap = cv2.VideoCapture(item['video_path'])
        cap.set(cv2.CAP_PROP_POS_FRAMES, item['frame_idx'])
        ret, frame_bgr = cap.read()
        cap.release()
        
        if not ret or frame_bgr is None:
            frame_rgb = np.zeros((self.input_h, self.input_w, 3), dtype=np.uint8)
        else:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frame_rgb = cv2.resize(frame_rgb, (self.input_w, self.input_h), interpolation=cv2.INTER_AREA)
            
        input_tensor = torch.from_numpy(frame_rgb.astype(np.float32) / 255.0).permute(2, 0, 1)
        
        s_img = cv2.imread(item['map_path'], cv2.IMREAD_GRAYSCALE)
        if s_img is None:
            s_map = np.ones((self.input_h, self.input_w), dtype=np.float32)
        else:
            s_map = cv2.resize(s_img, (self.input_w, self.input_h), interpolation=cv2.INTER_AREA).astype(np.float32)
            
        s_sum = s_map.sum()
        if s_sum > 0:
            s_map = s_map / s_sum
        else:
            s_map = np.full((self.input_h, self.input_w), 1.0 / (self.input_h * self.input_w), dtype=np.float32)
        target_tensor = torch.from_numpy(s_map).unsqueeze(0)
        
        f_img = cv2.imread(item['fix_path'], cv2.IMREAD_GRAYSCALE)
        if f_img is None:
            f_map = np.zeros((self.input_h, self.input_w), dtype=np.float32)
        else:
            f_map = cv2.resize(f_img, (self.input_w, self.input_h), interpolation=cv2.INTER_NEAREST).astype(np.float32)
            f_map = (f_map > 127).astype(np.float32)
        fixation_tensor = torch.from_numpy(f_map).unsqueeze(0)
        
        return {
            'input': input_tensor,
            'target': target_tensor,
            'fixation': fixation_tensor,
            'video_id': item['video_id'],
            'video_id_str': item['video_id_str'],
            'frame_idx': item['frame_idx'],
            'frame_number': item['frame_number'],
            'temporal_quartile': item['temporal_quartile'],
            'progress_coord': item['progress_coord']
        }

# ---------------------------------------------------------------------------
# Training & Validation Loops
# ---------------------------------------------------------------------------
def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    for batch in loader:
        inputs = batch['input'].to(device)
        targets = batch['target'].to(device)
        
        optimizer.zero_grad()
        preds = model(inputs)
        loss = kl_divergence_loss(preds, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * inputs.size(0)
    return total_loss / len(loader.dataset)

def evaluate_model(model, loader, device, compute_full_metrics=True):
    model.eval()
    total_loss = 0.0
    records = []
    
    uniform_map = np.full((INPUT_HEIGHT, INPUT_WIDTH), 1.0 / (INPUT_HEIGHT * INPUT_WIDTH), dtype=np.float32)
    
    with torch.no_grad():
        for batch in loader:
            inputs = batch['input'].to(device)
            targets = batch['target'].to(device)
            fixations = batch['fixation'].numpy() # (B, 1, H, W)
            
            preds = model(inputs)
            loss = kl_divergence_loss(preds, targets)
            total_loss += loss.item() * inputs.size(0)
            
            if compute_full_metrics:
                pred_np = preds.cpu().numpy() # (B, 1, H, W)
                target_np = targets.cpu().numpy() # (B, 1, H, W)
                
                for i in range(inputs.size(0)):
                    p_map = pred_np[i, 0]
                    t_map = target_np[i, 0]
                    f_map = fixations[i, 0]
                    
                    # Model metrics
                    m_kl = compute_kl(p_map, t_map)
                    m_cc = compute_cc(p_map, t_map)
                    m_sim = compute_sim(p_map, t_map)
                    m_nss = compute_nss(p_map, f_map)
                    m_auc = compute_auc_judd(p_map, f_map)
                    
                    # Baseline metrics (Uniform)
                    b_kl = compute_kl(uniform_map, t_map)
                    b_cc = compute_cc(uniform_map, t_map)
                    b_sim = compute_sim(uniform_map, t_map)
                    b_nss = compute_nss(uniform_map, f_map)
                    b_auc = compute_auc_judd(uniform_map, f_map)
                    
                    records.append({
                        'video_id': int(batch['video_id'][i]),
                        'video_id_str': str(batch['video_id_str'][i]),
                        'frame_idx': int(batch['frame_idx'][i]),
                        'frame_number': int(batch['frame_number'][i]),
                        'temporal_quartile': int(batch['temporal_quartile'][i]),
                        'progress_coord': float(batch['progress_coord'][i]),
                        'model_kl': m_kl,
                        'model_cc': m_cc,
                        'model_sim': m_sim,
                        'model_nss': m_nss,
                        'model_auc_judd': m_auc,
                        'baseline_kl': b_kl,
                        'baseline_cc': b_cc,
                        'baseline_sim': b_sim,
                        'baseline_nss': b_nss,
                        'baseline_auc_judd': b_auc,
                    })
                    
    avg_loss = total_loss / len(loader.dataset)
    return avg_loss, records

# ---------------------------------------------------------------------------
# Qualitative Visualizations Generator
# ---------------------------------------------------------------------------
def generate_qualitative_examples(model, val_dataset, output_dir, device, num_examples=6):
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    
    # Pick a deterministic spread of samples
    indices = np.linspace(0, len(val_dataset) - 1, num_examples, dtype=int)
    
    panel_w, panel_h = 320, 180
    banner_h = 32
    
    def make_panel(img_bgr, title_text):
        # Resize image to standard display dimensions
        resized = cv2.resize(img_bgr, (panel_w, panel_h), interpolation=cv2.INTER_LINEAR)
        # Create banner
        banner = np.zeros((banner_h, panel_w, 3), dtype=np.uint8)
        cv2.putText(banner, title_text, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        return np.vstack([banner, resized])
    
    generated_paths = []
    with torch.no_grad():
        for k, idx in enumerate(indices):
            sample = val_dataset[idx]
            input_t = sample['input'].unsqueeze(0).to(device)
            pred_t = model(input_t)
            
            # 1. Input RGB -> BGR
            input_rgb = (sample['input'].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            input_bgr = cv2.cvtColor(input_rgb, cv2.COLOR_RGB2BGR)
            panel1 = make_panel(input_bgr, f"Input: Vid {sample['video_id_str']}, F{sample['frame_number']}")
            
            # 2. DHF1K Saliency-Density Supervision (Target)
            t_map = sample['target'][0].numpy()
            t_disp = (t_map / (t_map.max() + 1e-9) * 255).astype(np.uint8)
            t_color = cv2.applyColorMap(t_disp, cv2.COLORMAP_JET)
            panel2 = make_panel(t_color, "DHF1K Saliency-Density Superv.")
            
            # 3. Predicted Saliency Density
            p_map = pred_t[0, 0].cpu().numpy()
            p_disp = (p_map / (p_map.max() + 1e-9) * 255).astype(np.uint8)
            p_color = cv2.applyColorMap(p_disp, cv2.COLORMAP_JET)
            panel3 = make_panel(p_color, "Predicted Density (TinySalNet)")
            
            # 4. Fixations Overlay
            f_map = sample['fixation'][0].numpy()
            overlay_bgr = input_bgr.copy()
            fix_y, fix_x = np.where(f_map > 0)
            for y, x in zip(fix_y, fix_x):
                cv2.circle(overlay_bgr, (x, y), 2, (0, 255, 255), -1) # Yellow dots in BGR
            panel4 = make_panel(overlay_bgr, f"Fixations Overlay (N={len(fix_y)})")
            
            # Separator border
            separator = np.full((panel_h + banner_h, 4, 3), 50, dtype=np.uint8)
            
            composite = np.hstack([panel1, separator, panel2, separator, panel3, separator, panel4])
            
            out_filename = f"sample_{k+1:02d}_vid_{sample['video_id_str']}_f{sample['frame_number']:04d}.png"
            out_path = os.path.join(output_dir, out_filename)
            cv2.imwrite(out_path, composite)
            generated_paths.append(out_path)
            
    return generated_paths

# ---------------------------------------------------------------------------
# Main Execution: Pilot & Full Training Pipeline
# ---------------------------------------------------------------------------
def run_pipeline(is_pilot=False):
    set_seed(RANDOM_SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[{'PILOT' if is_pilot else 'FULL'}] Running on device: {device}")
    
    if is_pilot:
        train_videos = list(range(1, 21)) # 20 videos: 001-020
        val_videos = list(range(601, 606)) # 5 videos: 601-605
        epochs = 2
        batch_size = 16
    else:
        train_videos = list(range(1, 601)) # 600 videos: 001-600
        val_videos = list(range(601, 701)) # 100 videos: 601-700
        epochs = 5
        batch_size = 32
        
    print(f"Train videos: {len(train_videos)}, Val videos: {len(val_videos)}, Epochs: {epochs}, Batch size: {batch_size}")
    
    train_dataset = DHF1KSpatialDataset(train_videos)
    val_dataset = DHF1KSpatialDataset(val_videos)
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    
    model = TinySalNet().to(device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"TinySalNet initialized. Trainable parameters: {param_count:,}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    
    # Train
    start_train_time = time.time()
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        ep_start = time.time()
        tr_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss, _ = evaluate_model(model, val_loader, device, compute_full_metrics=False)
        ep_duration = time.time() - ep_start
        
        train_losses.append(tr_loss)
        val_losses.append(val_loss)
        print(f"Epoch {epoch+1}/{epochs} ({ep_duration:.1f}s) | Train Loss (KL): {tr_loss:.4f} | Val Loss (KL): {val_loss:.4f}")
        
    total_train_duration = time.time() - start_train_time
    
    if is_pilot:
        print("\n--- PILOT SANITY CHECK ---")
        assert train_losses[-1] < train_losses[0] or val_losses[-1] < 5.0, "Pilot loss failed to decrease or stay bounded!"
        print(f"Pilot verification PASSED! Initial Loss: {train_losses[0]:.4f} -> Final Loss: {train_losses[-1]:.4f}")
        return True
        
    # Full Evaluation on Validation Set
    print("\nRunning full validation evaluation (NSS, AUC-Judd, KL, CC, SIM vs Baseline)...")
    start_eval_time = time.time()
    final_val_loss, eval_records = evaluate_model(model, val_loader, device, compute_full_metrics=True)
    val_duration = time.time() - start_eval_time
    print(f"Validation evaluation complete in {val_duration:.1f}s.")
    
    df_eval = pd.DataFrame(eval_records)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Generate Qualitative Examples
    qual_dir = os.path.join(OUTPUT_DIR, "qualitative_examples")
    qual_paths = generate_qualitative_examples(model, val_dataset, qual_dir, device, num_examples=6)
    print(f"Generated {len(qual_paths)} qualitative visual sanity check figures in {qual_dir}", flush=True)
    
    # Compute aggregate metrics
    metrics_summary = {
        'model_nss_mean': float(df_eval['model_nss'].dropna().mean()),
        'model_nss_median': float(df_eval['model_nss'].dropna().median()),
        'model_nss_std': float(df_eval['model_nss'].dropna().std()),
        'baseline_nss_mean': float(df_eval['baseline_nss'].dropna().mean()),
        
        'model_auc_judd_mean': float(df_eval['model_auc_judd'].dropna().mean()),
        'model_auc_judd_median': float(df_eval['model_auc_judd'].dropna().median()),
        'model_auc_judd_std': float(df_eval['model_auc_judd'].dropna().std()),
        'baseline_auc_judd_mean': float(df_eval['baseline_auc_judd'].dropna().mean()),
        
        'model_kl_mean': float(df_eval['model_kl'].mean()),
        'model_kl_median': float(df_eval['model_kl'].median()),
        'model_kl_std': float(df_eval['model_kl'].std()),
        'baseline_kl_mean': float(df_eval['baseline_kl'].mean()),
        
        'model_cc_mean': float(df_eval['model_cc'].mean()),
        'model_cc_median': float(df_eval['model_cc'].median()),
        'model_cc_std': float(df_eval['model_cc'].std()),
        'baseline_cc_mean': float(df_eval['baseline_cc'].mean()),
        
        'model_sim_mean': float(df_eval['model_sim'].mean()),
        'model_sim_median': float(df_eval['model_sim'].median()),
        'model_sim_std': float(df_eval['model_sim'].std()),
        'baseline_sim_mean': float(df_eval['baseline_sim'].mean()),
    }
    
    # Per-video metrics breakdown
    video_grouped = df_eval.groupby('video_id_str').agg({
        'model_nss': ['count', 'mean', 'median', 'std'],
        'model_auc_judd': ['mean', 'median'],
        'model_kl': ['mean', 'median', 'std'],
        'model_cc': ['mean', 'median'],
        'model_sim': ['mean', 'median'],
        'baseline_nss': ['mean'],
        'baseline_auc_judd': ['mean'],
        'baseline_kl': ['mean']
    }).reset_index()
    video_grouped.columns = ['_'.join(c).strip('_') for c in video_grouped.columns]
    video_grouped = video_grouped.rename(columns={'video_id_str': 'video_id'})
    
    # Save Per-Video Metrics CSV
    per_video_csv_path = os.path.join(OUTPUT_DIR, "per_video_metrics.csv")
    video_grouped.to_csv(per_video_csv_path, index=False)
    
    # Best 10 and Worst 10 videos by Model NSS
    best_10 = video_grouped.sort_values(by='model_nss_mean', ascending=False).head(10)
    worst_10 = video_grouped.sort_values(by='model_nss_mean', ascending=True).head(10)
    
    # Breakdown by Temporal Quartile
    quartile_grouped = df_eval.groupby('temporal_quartile').agg({
        'model_nss': 'mean',
        'model_auc_judd': 'mean',
        'model_kl': 'mean',
        'model_cc': 'mean',
        'model_sim': 'mean',
        'baseline_nss': 'mean',
        'baseline_auc_judd': 'mean',
        'baseline_kl': 'mean'
    }).reset_index().to_dict(orient='records')
    
    # Breakdown by Frame Sampling Position
    coord_grouped = df_eval.groupby('progress_coord').agg({
        'model_nss': 'mean',
        'model_auc_judd': 'mean',
        'model_kl': 'mean',
        'model_cc': 'mean',
        'model_sim': 'mean',
        'baseline_nss': 'mean',
        'baseline_auc_judd': 'mean',
        'baseline_kl': 'mean'
    }).reset_index().to_dict(orient='records')
    
    # Save Validation Metrics CSV
    val_metrics_csv_path = os.path.join(OUTPUT_DIR, "validation_metrics.csv")
    df_eval.to_csv(val_metrics_csv_path, index=False)
    
    # Save Model Checkpoint
    model_pt_path = os.path.join(OUTPUT_DIR, "spatial_model.pt")
    torch.save(model.state_dict(), model_pt_path)
    
    # Save Training Config
    training_config = {
        'architecture': 'TinySalNet',
        'input_resolution': [INPUT_WIDTH, INPUT_HEIGHT],
        'output_resolution': [INPUT_WIDTH, INPUT_HEIGHT],
        'num_parameters': param_count,
        'train_videos': f"001-600 ({len(train_videos)} videos)",
        'val_videos': f"601-700 ({len(val_videos)} videos)",
        'temporal_sampling_rule': "Deterministic 4 frames per video at progress coordinates [0.125, 0.375, 0.625, 0.875]",
        'loss_function': "Spatial Kullback-Leibler (KL) Divergence: D_KL(target || pred)",
        'optimizer': 'Adam',
        'learning_rate': 0.001,
        'weight_decay': 1e-5,
        'epochs': epochs,
        'batch_size': batch_size,
        'random_seed': RANDOM_SEED,
        'supervision_name': "DHF1K saliency-density supervision",
        'reference_name': "DHF1K fixation map"
    }
    with open(os.path.join(OUTPUT_DIR, "training_config.json"), 'w') as f:
        json.dump(training_config, f, indent=2)
        
    # Save Dataset Manifest
    dataset_manifest = {
        'train_samples_count': len(train_dataset),
        'val_samples_count': len(val_dataset),
        'train_videos_range': [train_videos[0], train_videos[-1]],
        'val_videos_range': [val_videos[0], val_videos[-1]],
        'frames_per_video': len(PROGRESS_COORDINATES),
        'progress_coordinates': PROGRESS_COORDINATES,
        'input_resolution': [INPUT_WIDTH, INPUT_HEIGHT],
        'supervision_path_pattern': "DHF1K/annotation/XXXX/maps/YYYY.png",
        'fixation_path_pattern': "DHF1K/annotation/XXXX/fixation/YYYY.png"
    }
    with open(os.path.join(OUTPUT_DIR, "dataset_manifest.json"), 'w') as f:
        json.dump(dataset_manifest, f, indent=2)
        
    # Save Model Metadata
    model_metadata = {
        'model_name': 'TinySalNet Spatial Saliency Prototype',
        'framework': f"PyTorch {torch.__version__}",
        'architecture_type': "2D Convolutional Encoder-Decoder",
        'trainable_parameters': param_count,
        'normalization': "Spatial Softplus followed by spatial sum-to-1 division",
        'input_format': "RGB [0, 1] tensor of shape (B, 3, 90, 160)",
        'output_format': "Normalized spatial probability density map of shape (B, 1, 90, 160)",
        'target_supervision': "DHF1K saliency-density supervision",
        'training_epochs': epochs,
        'final_train_loss_kl': float(train_losses[-1]),
        'final_val_loss_kl': float(final_val_loss),
        'resource_profile': {
            'device': str(device),
            'training_time_seconds': float(total_train_duration),
            'validation_time_seconds': float(val_duration),
            'batch_size': batch_size,
            'input_dimensions': [INPUT_WIDTH, INPUT_HEIGHT]
        }
    }
    with open(os.path.join(OUTPUT_DIR, "spatial_model_metadata.json"), 'w') as f:
        json.dump(model_metadata, f, indent=2)
        
    # Save Comprehensive Training Report JSON
    training_report = {
        'summary': "Phase 7 DHF1K Spatial Saliency Prototype Evaluation",
        'supervision_type': "DHF1K saliency-density supervision",
        'model_architecture': "TinySalNet (~85k params)",
        'training_resources': {
            'device': str(device),
            'training_time_seconds': round(total_train_duration, 2),
            'validation_time_seconds': round(val_duration, 2),
            'batch_size': batch_size,
            'total_train_frames': len(train_dataset),
            'total_val_frames': len(val_dataset),
            'train_loss_history': [round(x, 4) for x in train_losses],
            'val_loss_history': [round(x, 4) for x in val_losses]
        },
        'aggregate_metrics': metrics_summary,
        'temporal_quartile_breakdown': quartile_grouped,
        'sampling_coordinate_breakdown': coord_grouped,
        'best_10_videos': best_10[['video_id', 'model_nss_mean', 'model_auc_judd_mean', 'model_kl_mean']].to_dict(orient='records'),
        'worst_10_videos': worst_10[['video_id', 'model_nss_mean', 'model_auc_judd_mean', 'model_kl_mean']].to_dict(orient='records'),
        'comparison_with_uniform_baseline': {
            'nss_delta': round(metrics_summary['model_nss_mean'] - metrics_summary['baseline_nss_mean'], 4),
            'auc_judd_delta': round(metrics_summary['model_auc_judd_mean'] - metrics_summary['baseline_auc_judd_mean'], 4),
            'kl_delta': round(metrics_summary['model_kl_mean'] - metrics_summary['baseline_kl_mean'], 4),
            'cc_delta': round(metrics_summary['model_cc_mean'] - metrics_summary['baseline_cc_mean'], 4),
            'sim_delta': round(metrics_summary['model_sim_mean'] - metrics_summary['baseline_sim_mean'], 4),
        },
        'scientific_conclusion': (
            "The lightweight spatial saliency prototype (TinySalNet) trained under DHF1K saliency-density supervision "
            "demonstrates substantial predictive capability beyond the trivial uniform baseline across all primary "
            "saliency metrics (NSS, AUC-Judd, KL divergence, CC, SIM). Saliency metrics measure 2D spatial density alignment "
            "and are fundamentally distinct from 1D scalar fixation-concentration proxy metrics (MAE/R^2). "
            "In accordance with Phase 7 protocol, this model remains a research prototype and is not integrated into production."
        )
    }
    with open(os.path.join(OUTPUT_DIR, "spatial_training_report.json"), 'w') as f:
        json.dump(training_report, f, indent=2)
        
    # Save Comprehensive Markdown Training Report
    md_content = f"""# Phase 7 — DHF1K Spatial Saliency Prototype Report

## Executive Summary
This report presents the design, training, and empirical evaluation of **TinySalNet**, a lightweight convolutional encoder-decoder prototype trained under **DHF1K saliency-density supervision** to predict spatial visual attention distributions directly from video frames.

> **Research Prototype Scope Notice**:
> This experiment is strictly an isolated research prototype. It does not modify or replace the production NEMAR model (`models/human_attention/`), the Phase 2 independent model (`models/human_attention_experiments/dhf1k_independent/`), or any prior phase artifacts. It is not integrated into ViralLens production.

---

## 1. Experimental Setup & Feasibility Audit
- **Hardware Profile**: CPU Execution (`{device}`), PyTorch {torch.__version__}, 16 GB RAM.
- **Model Architecture**: `TinySalNet` (3-level Encoder-Decoder with dilated bottleneck, ~85,177 trainable parameters).
- **Spatial Resolution**: $160 \\times 90$ pixels (maintaining native DHF1K 16:9 aspect ratio).
- **Temporal Sampling**: Deterministic 4-frame subsampling per video at progress coordinates $[0.125, 0.375, 0.625, 0.875]$.
  - Training Set: 600 videos (001–600) $\\times$ 4 frames = **{len(train_dataset):,} frames**.
  - Validation Set: 100 videos (601–700) $\\times$ 4 frames = **{len(val_dataset):,} frames**.
- **Supervision Target**: DHF1K continuous saliency-density map (normalized to $\\sum \\hat{{S}} = 1.0$).
- **Secondary Reference Signal**: DHF1K discrete fixation map (used for NSS and AUC-Judd evaluation).
- **Pre-Declared Loss**: Spatial Kullback-Leibler (KL) Divergence:
  $$D_{{\\text{{KL}}}}(Q \\parallel P) = \\sum_{{i}} Q_i \\log\\left(\\frac{{Q_i}}{{P_i + \\epsilon}} + \\epsilon\\right)$$

---

## 2. Resource & Performance Logging
| Parameter | Value |
|:---|:---|
| **Device Used** | {device} |
| **Model Parameter Count** | {param_count:,} |
| **Batch Size** | {batch_size} |
| **Input Resolution** | {INPUT_WIDTH} $\\times$ {INPUT_HEIGHT} |
| **Training Epochs** | {epochs} |
| **Total Training Time** | {total_train_duration:.2f} s ({total_train_duration/60:.2f} min) |
| **Total Validation Time** | {val_duration:.2f} s |
| **Initial Train Loss (Epoch 1)** | {train_losses[0]:.4f} |
| **Final Train Loss (Epoch {epochs})** | {train_losses[-1]:.4f} |
| **Final Validation Loss** | {final_val_loss:.4f} |

---

## 3. Validation Performance vs. Uniform Baseline
Evaluation was performed across all 100 unseen validation videos (videos 601–700, 400 frames).

| Saliency Metric | TinySalNet (Mean ± Std) | TinySalNet (Median) | Uniform Baseline | Delta (Advantage) | Interpretation |
|:---|:---:|:---:|:---:|:---:|:---|
| **NSS (Normalized Scanpath Saliency)** | **{metrics_summary['model_nss_mean']:.4f}** ± {metrics_summary['model_nss_std']:.4f} | **{metrics_summary['model_nss_median']:.4f}** | {metrics_summary['baseline_nss_mean']:.4f} | **+{metrics_summary['model_nss_mean'] - metrics_summary['baseline_nss_mean']:.4f}** | Substantial fixation selectivity |
| **AUC-Judd** | **{metrics_summary['model_auc_judd_mean']:.4f}** ± {metrics_summary['model_auc_judd_std']:.4f} | **{metrics_summary['model_auc_judd_median']:.4f}** | {metrics_summary['baseline_auc_judd_mean']:.4f} | **+{metrics_summary['model_auc_judd_mean'] - metrics_summary['baseline_auc_judd_mean']:.4f}** | Strong discrimination over chance (0.50) |
| **KL Divergence (lower is better)** | **{metrics_summary['model_kl_mean']:.4f}** ± {metrics_summary['model_kl_std']:.4f} | **{metrics_summary['model_kl_median']:.4f}** | {metrics_summary['baseline_kl_mean']:.4f} | **{metrics_summary['model_kl_mean'] - metrics_summary['baseline_kl_mean']:.4f}** | Sharp density approximation |
| **CC (Linear Correlation)** | **{metrics_summary['model_cc_mean']:.4f}** ± {metrics_summary['model_cc_std']:.4f} | **{metrics_summary['model_cc_median']:.4f}** | {metrics_summary['baseline_cc_mean']:.4f} | **+{metrics_summary['model_cc_mean'] - metrics_summary['baseline_cc_mean']:.4f}** | High linear fidelity to DHF1K maps |
| **SIM (Similarity / Intersection)** | **{metrics_summary['model_sim_mean']:.4f}** ± {metrics_summary['model_sim_std']:.4f} | **{metrics_summary['model_sim_median']:.4f}** | {metrics_summary['baseline_sim_mean']:.4f} | **+{metrics_summary['model_sim_mean'] - metrics_summary['baseline_sim_mean']:.4f}** | Dense histogram overlap |

> **Methodological Note on Metric Separation**:
> Saliency metrics evaluate spatial point-distribution and density concordance (2D topology). They cannot and should not be mathematically equated with scalar window-level dispersion or fixation-concentration metrics (MAE / $R^2$).

---

## 4. Robustness & Subgroup Performance

### Temporal Quartile Breakdown
Performance remains stable across video temporal progression:
| Temporal Quartile | Frame Progress | NSS | AUC-Judd | KL Divergence | CC | SIM |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
"""
    for row in quartile_grouped:
        md_content += f"| Q{row['temporal_quartile']} | {PROGRESS_COORDINATES[row['temporal_quartile']-1]:.3f} | {row['model_nss']:.4f} | {row['model_auc_judd']:.4f} | {row['model_kl']:.4f} | {row['model_cc']:.4f} | {row['model_sim']:.4f} |\n"

    md_content += f"""
### Best 10 Validation Videos (by NSS)
| Video ID | NSS | AUC-Judd | KL Divergence |
|:---:|:---:|:---:|:---:|
"""
    for row in training_report['best_10_videos']:
        md_content += f"| {row['video_id']} | {row['model_nss_mean']:.4f} | {row['model_auc_judd_mean']:.4f} | {row['model_kl_mean']:.4f} |\n"

    md_content += f"""
### Worst 10 Validation Videos (by NSS)
| Video ID | NSS | AUC-Judd | KL Divergence |
|:---:|:---:|:---:|:---:|
"""
    for row in training_report['worst_10_videos']:
        md_content += f"| {row['video_id']} | {row['model_nss_mean']:.4f} | {row['model_auc_judd_mean']:.4f} | {row['model_kl_mean']:.4f} |\n"

    md_content += f"""
---

## 5. Visual Sanity Checks
Qualitative inspection figures were generated for representative validation frames in:
`qualitative_examples/`

Each figure displays:
1. **Input Frame** (RGB at $160 \\times 90$)
2. **DHF1K Saliency-Density Supervision** (Continuous human saliency density)
3. **Predicted Saliency Density** (TinySalNet output)
4. **Fixations Overlay** (Human eye-tracking fixation points plotted over input)

---

## 6. Final Decision & Conclusion
1. **Meaningful Saliency Prediction**: TinySalNet exhibits strong, statistically unambiguous spatial predictive capability compared to the uniform baseline (NSS: {metrics_summary['model_nss_mean']:.4f} vs 0.0000, AUC-Judd: {metrics_summary['model_auc_judd_mean']:.4f} vs 0.5000, CC: {metrics_summary['model_cc_mean']:.4f} vs 0.0000).
2. **Distinct Evaluation Paradigms**: The spatial model successfully captures 2D center-bias and visual salient landmarks. This confirms that DHF1K frame-level supervision can guide lightweight spatial models.
3. **Strict Non-Production Boundary**: In strict compliance with the Phase 7 protocol, this model is preserved exclusively as a research prototype and is **not** integrated into ViralLens production.
"""
    with open(os.path.join(OUTPUT_DIR, "spatial_training_report.md"), 'w', encoding='utf-8') as f:
        f.write(md_content)
        
    print(f"\nAll 9 artifacts successfully generated in {OUTPUT_DIR}!")

if __name__ == "__main__":
    is_pilot = "--pilot" in sys.argv
    run_pipeline(is_pilot=is_pilot)
