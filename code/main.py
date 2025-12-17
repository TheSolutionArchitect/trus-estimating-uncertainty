
import os
import sys
import json
import math
import glob
import argparse
import traceback
from datetime import datetime
import random
import itertools
import time
import gc
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold

from monai.data import Dataset
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Resized,
    ToTensord,
    MapTransform,
    RandRotate90d,
    RandFlipd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandZoomd,
    RandAdjustContrastd,
    ScaleIntensityRanged,
)
from monai.losses import DiceLoss, FocalLoss, TverskyLoss
from monai.networks.nets import SwinUNETR

from scipy.ndimage import label as cc_label, center_of_mass
from scipy.optimize import linear_sum_assignment
from scipy.stats import t as student_t

import logging
import logging.handlers

# Matplotlib for collages and optional uncertainty maps
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib import cm as mpl_cm


# ----
# Config and Logging
# ----

class MilestoneFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:  # type: ignore[override]
        return bool(getattr(record, "milestone", False))


def _strip_json_comments_and_trailing_commas(text: str) -> str:
    import re
    text = re.sub(r"//.*", "", text)
    text = re.sub(r"/\*[\s\S]*?\*/", "", text)
    text = re.sub(r",(\s*[}\]])", r"\1", text)
    return text


def load_config_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        cleaned = _strip_json_comments_and_trailing_commas(raw)
        return json.loads(cleaned)


class Config:
    def __init__(self, d: dict):
        try:
            self.model_name = d["model_name"]
            self.paths = d["paths"]
            self.train = d["train"]
            self.metrics = d["metrics"]
            self.cv = d["cv"]
            self.viz = d["viz"]
        except KeyError as e:
            raise ValueError(f"Missing required config section or key: {e}")

        self.description = d.get("description", "")

        self.loss = d.get(
            "loss",
            {
                "focal_gamma": 3.0,
                "tversky_alpha": 0.6,
                "tversky_beta": 0.4,
                "combined_weights": [0.1, 0.7, 0.2],
            },
        )

        self.model = d.get(
            "model",
            {
                "arch": "SwinUNETR",
                "in_channels": 3,
                "out_channels": 1,
                "feature_size": 48,
                "drop_rate": 0.1,
                "spatial_dims": 2,
                "use_checkpoint": True,
            },
        )
        self.model.setdefault("arch", "SwinUNETR")
        self.model.setdefault("in_channels", 3)
        self.model.setdefault("out_channels", 1)
        self.model.setdefault("feature_size", 48)
        self.model.setdefault("drop_rate", 0.1)
        self.model.setdefault("spatial_dims", 2)
        self.model.setdefault("use_checkpoint", True)

        self.augment = d.get(
            "augment",
            {
                "rotate90_prob": 0.5,
                "flip_prob": 0.5,
                "flip_axis": 0,
                "zoom_prob": 0.5,
                "min_zoom": 0.7,
                "max_zoom": 1.3,
                "gauss_noise_prob": 0.3,
                "gauss_noise_std": 0.01,
                "gauss_smooth_prob": 0.3,
                "gauss_sigma_x_min": 0.5,
                "gauss_sigma_x_max": 1.0,
                "adjust_contrast_prob": 0.3,
                "gamma_min": 0.7,
                "gamma_max": 1.3,
            },
        )

        # Train defaults
        self.train.setdefault("use_amp", True)
        self.train.setdefault("micro_batch_size", None)
        self.train.setdefault("auto_microbatch_on_oom", True)
        self.train.setdefault("max_auto_microbatch_halvings", 3)
        self.train.setdefault("num_workers", 0)
        self.train.setdefault("plateau_reduce_factor", 0.5)
        self.train.setdefault("plateau_patience_epochs", 3)
        self.train.setdefault("early_stop_min_delta_mm", 0.0)

        # Search defaults (Optuna only)
        default_search = {
            "enabled": True,
            "strategy": "bayes_opt",
            "bayes_opt": {"n_trials": 15, "seed": 42, "param_space": {}},
            "cv_n_splits": 5,
            "cv_n_repeats": 1,
            "cv_max_epochs": 50,
            "scoring": {"primary": "f1", "tie_breaker": "mean_ed_mm"},
        }
        self.search = default_search
        if "search" in d:
            self.search.update(d["search"])
            if "bayes_opt" in d["search"] and isinstance(d["search"]["bayes_opt"], dict):
                self.search["bayes_opt"].update(d["search"]["bayes_opt"])

        # Uncertainty defaults
        self.uncertainty = d.get(
            "uncertainty",
            {
                "enabled": True,
                "mc_dropout": {"enabled": True, "num_samples": 15, "seed": 12345},
                "predictive_entropy": {"enabled": True},
                "epistemic": {"enabled": True, "measure": "mi"},  # "mi" or "variance"
                "per_needle_pool": {
                    "window_size": 11,
                    "pool_type": "mean",     # "mean" or "max"
                    "normalize": "minmax_slice"  # "minmax_slice" or "none"
                },
                "combine": {
                    "enabled": True,
                    "weight_pe": 0.5,
                    "weight_epi": 0.5,
                    "normalization": "minmax_slice"  # map-level normalization for combined map
                },
                "save_maps": {
                    "enabled": True,
                    "save_pe": True,
                    "save_epi": True,
                    "save_combined": True,
                    "format": "png",
                    "colormap": "magma",
                    "colorbar": True
                }
            },
        )

        # Basic path checks
        for k in ["input_dir", "mask_dir", "output_dir"]:
            if k not in self.paths or not self.paths[k]:
                raise ValueError(f"Missing or empty paths.{k} in config")

        # Required keys
        for k in ["img_size", "batch_size", "accumulation_steps", "max_epochs", "patience", "learning_rate", "class_weight"]:
            if k not in self.train:
                raise ValueError(f"Missing train.{k} in config")
        for k in ["pred_threshold", "min_component_size", "pixel_to_mm", "detection_tolerance_mm"]:
            if k not in self.metrics:
                raise ValueError(f"Missing metrics.{k} in config")
        for k in ["k_folds", "random_state", "test_size", "evaluation_fold"]:
            if k not in self.cv:
                raise ValueError(f"Missing cv.{k} in config")
        for k in ["save_val_vis", "max_val_vis_per_epoch"]:
            if k not in self.viz:
                raise ValueError(f"Missing viz.{k} in config")

        # Viz defaults
        self.viz.setdefault("rotate_cw_90", False)
        self.viz.setdefault("overlay_orientation", "none")
        self.viz.setdefault("save_test_predictions", True)
        self.viz.setdefault("test_pred_draw_radius", 4)
        self.viz.setdefault("test_predictions_max_slices_per_patient", None)
        self.viz.setdefault("overlay_prob_on_test_images", True)
        self.viz.setdefault("prob_digits", 2)
        self.viz.setdefault("prob_text_color", [255, 255, 0])
        self.viz.setdefault("prob_font_size", 14)
        self.viz.setdefault("prob_font_path", None)
        self.viz.setdefault("prob_label_prefix", "")
        self.viz.setdefault("collage_dpi", 140)
        self.viz.setdefault("collage_figsize", [12, 10])
        self.viz.setdefault("collage_title_fontsize", 12)
        self.viz.setdefault("pred_point_color", [0, 1, 0])
        self.viz.setdefault("pred_point_edgecolor", [1, 1, 0])
        self.viz.setdefault("pred_point_radius_px", 7)
        self.viz.setdefault("pred_point_linewidth", 1.5)
        # New viz toggles for uncertainty composite (3x2)
        self.viz.setdefault("save_uncertainty_composite", True)
        self.viz.setdefault("uncertainty_composite_figsize", [12, 14])
        self.viz.setdefault("uncertainty_composite_dpi", 140)
        self.viz.setdefault("uncertainty_title_fontsize", 12)

        # Metric defaults
        self.metrics.setdefault("prob_value_method", "centroid")
        self.metrics.setdefault("tolerance_sweep_mm", [])
        self.metrics.setdefault("ci_confidence_level", 0.95)

        # Outputs
        self.timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.base_output_dir = os.path.join(self.paths["output_dir"], self.model_name, self.timestamp)
        self.log_dir = os.path.join(self.base_output_dir, "log")
        self.coords_dir = os.path.join(self.base_output_dir, "needle_coordinates")
        self.vis3d_dir = os.path.join(self.base_output_dir, "3d_visualizations")
        self.epoch_csv_dir = os.path.join(self.base_output_dir, "per_epoch_metrics")
        self.folds_json_dir = os.path.join(self.base_output_dir, "fold_hyperparams")
        self.search_dir = os.path.join(self.base_output_dir, "search")
        self.predictions_dir = os.path.join(self.base_output_dir, "predictions")
        for p in [self.base_output_dir, self.log_dir, self.coords_dir, self.vis3d_dir, self.epoch_csv_dir, self.folds_json_dir, self.search_dir, self.predictions_dir]:
            os.makedirs(p, exist_ok=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get(self, dotted: str, default: Any = None) -> Any:
        cur: Any = self.__dict__
        for part in dotted.split("."):
            if isinstance(cur, dict) and part in cur:
                cur = cur[part]
            else:
                return default
        return cur


def setup_logger(cfg: Config) -> logging.Logger:
    logger = logging.getLogger("needle_train")
    logger.setLevel(logging.INFO)

    log_path = os.path.join(cfg.log_dir, f"log-{cfg.timestamp}.log")
    file_handler = logging.handlers.RotatingFileHandler(log_path, maxBytes=200*1024*1024, backupCount=3)
    formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s", "%Y-%m-%d %H:%M:%S")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    console.setLevel(logging.INFO)
    console.addFilter(MilestoneFilter())

    logger.handlers = []
    logger.addHandler(file_handler)
    logger.addHandler(console)
    return logger


def install_exception_hook(logger: logging.Logger) -> None:
    def exception_hook(exctype, value, tb):
        logger.error("Uncaught exception: %s", str(value))
        logger.error("Type: %s", str(exctype))
        logger.error("Traceback:\n" + "".join(traceback.format_tb(tb)))
        print(f"Uncaught exception: {str(value)}")
        sys.__excepthook__(exctype, value, tb)
    sys.excepthook = exception_hook


# ----
# Data utilities
# ----

class EnsureBinaryMaskCHW(MapTransform):
    def __init__(self, keys):
        super().__init__(keys)
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            x = np.asarray(d[key])
            x = np.squeeze(x)
            if x.ndim == 2:
                hw = x
            elif x.ndim == 3:
                ch_axis = 2 if x.shape[2] in (1, 3, 4) else 0
                hw = np.max(np.moveaxis(x, ch_axis, -1), axis=-1)
            else:
                z = np.squeeze(x)
                while z.ndim > 2:
                    z = z.max(axis=0)
                hw = z
            d[key] = (hw > 0).astype(np.float32)[None, :, :]
        return d


def _read_png_as_rgb(path: str) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def _read_png_as_gray(path: str) -> np.ndarray:
    return np.array(Image.open(path).convert("L"))


def get_patient_data_dicts(cfg: Config, logger: logging.Logger) -> Dict[str, List[dict]]:
    input_dir = os.path.abspath(cfg.paths["input_dir"])
    mask_dir = os.path.abspath(cfg.paths["mask_dir"])
    logger.info(f"Input dir: {input_dir}")
    logger.info(f"Mask dir:  {mask_dir}")

    if not os.path.isdir(input_dir) or not os.path.isdir(mask_dir):
        raise ValueError("Either input_dir or mask_dir does not exist.")

    patient_ids = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
    logger.info(f"Found {len(patient_ids)} patient folders.")

    patient_data: Dict[str, List[dict]] = {}
    skipped = []

    for pid in patient_ids:
        p_img_dir = os.path.join(input_dir, pid)
        p_msk_dir = os.path.join(mask_dir, pid)
        if not os.path.isdir(p_msk_dir):
            logger.warning(f"No mask folder for patient {pid}; skipping patient.")
            continue

        image_files = sorted(glob.glob(os.path.join(p_img_dir, "*.png")))
        slices = []
        for img_path in image_files:
            fname = os.path.basename(img_path)
            mask_path = os.path.join(p_msk_dir, fname)
            if not os.path.exists(mask_path):
                skipped.append((img_path, "Mask missing"))
                continue
            try:
                img_np = _read_png_as_rgb(img_path)
                _ = _read_png_as_gray(mask_path)
                if img_np.ndim != 3 or img_np.shape[2] != 3:
                    skipped.append((img_path, f"Not RGB, shape {img_np.shape}"))
                    continue
                try:
                    if "_slice_" in fname:
                        slice_num = int(fname.split("_slice_")[1].split(".")[0])
                    else:
                        slice_num = int(os.path.splitext(fname)[0].split("_")[-1])
                except Exception:
                    slice_num = 0
                slices.append(
                    {
                        "image": img_path,
                        "label": mask_path,
                        "image_path": img_path,
                        "label_path": mask_path,
                        "slice_num": slice_num,
                        "patient_id": pid,
                    }
                )
            except Exception as e:
                skipped.append((img_path, f"Error: {e}"))

        if slices:
            slices = sorted(slices, key=lambda x: x["slice_num"], reverse=True)
            patient_data[pid] = slices

    logger.info(f"Prepared data for {len(patient_data)} patients. Skipped {len(skipped)} items.")
    for p, r in skipped:
        logger.info(f"  {p}: {r}")
    logger.info(
        f"Dataset preparation completed: {len(patient_data)} patients, "
        f"{sum(len(v) for v in patient_data.values())} slices.",
        extra={"milestone": True},
    )
    return patient_data


def compute_patient_strat_labels(patient_data: dict) -> dict:
    labels = {}
    for pid, slices in patient_data.items():
        has_pos = 0
        for s in slices:
            m = _read_png_as_gray(s["label"])
            if np.any(m > 0):
                has_pos = 1
                break
        labels[pid] = has_pos
    return labels


# ----
# Needle utilities and metrics
# ----

def _centroids_from_mask(mask_np: np.ndarray, min_component_size: int):
    labeled, n = cc_label(mask_np.astype(np.uint8))
    cents, areas = [], []
    for i in range(1, n + 1):
        comp = (labeled == i)
        sz = int(comp.sum())
        if sz < min_component_size:
            continue
        y, x = center_of_mass(comp)
        cents.append((int(round(x)), int(round(y))))
        areas.append(sz)
    return cents, areas


def hungarian_one_to_one(gt_centroids, pred_centroids):
    if len(gt_centroids) == 0 or len(pred_centroids) == 0:
        return []
    G = len(gt_centroids)
    P = len(pred_centroids)
    dim = max(G, P)
    cost = np.zeros((dim, dim), dtype=np.float32) + 1e6
    for gi, (gx, gy) in enumerate(gt_centroids):
        for pi, (px, py) in enumerate(pred_centroids):
            d = math.hypot(px - gx, py - gy)
            cost[gi, pi] = d
    row_ind, col_ind = linear_sum_assignment(cost)
    pairs = []
    for r, c in zip(row_ind, col_ind):
        if r < G and c < P and cost[r, c] < 1e6:
            pairs.append((c, r))
    return pairs


def compute_euclidean_mm_and_matches(
    cfg: Config,
    outputs_bin: torch.Tensor,
    labels_bin: torch.Tensor,
    file_id: str,
    orig_hw: Tuple[Optional[int], Optional[int]],
    resized_hw: Tuple[int, int],
):
    pred_np = outputs_bin.squeeze().detach().cpu().numpy().astype(np.uint8)
    gt_np = labels_bin.squeeze().detach().cpu().numpy().astype(np.uint8)
    pred_centroids, _ = _centroids_from_mask(pred_np, cfg.metrics["min_component_size"])
    gt_centroids, _ = _centroids_from_mask(gt_np, cfg.metrics["min_component_size"])

    n_pred = len(pred_centroids)
    n_gt = len(gt_centroids)
    metrics = {"Needle_Count_Pred": n_pred, "Needle_Count_GT": n_gt}
    matches: List[Tuple[int, int]] = []

    if orig_hw[0] is None:
        mm_per_pixel = cfg.metrics["pixel_to_mm"]
    else:
        mm_per_pixel = cfg.metrics["pixel_to_mm"] * (orig_hw[0] / resized_hw[0])

    if n_pred == 0 or n_gt == 0:
        metrics.update(
            {
                "ED_mm_list": [],
                "Mean_Euclidean_Distance_mm": float("nan"),
                "Max_Euclidean_Distance_mm": float("nan"),
                "TP": 0,
                "FP": n_pred,
                "FN": n_gt,
            }
        )
        return metrics, matches, pred_centroids, gt_centroids

    pairs = hungarian_one_to_one(gt_centroids, pred_centroids)
    matches = pairs[:]

    dists_mm: List[float] = []
    for (pi, gi) in pairs:
        px, py = pred_centroids[pi]
        gx, gy = gt_centroids[gi]
        d_mm = math.hypot(px - gx, py - gy) * mm_per_pixel
        dists_mm.append(d_mm)

    tol = float(cfg.metrics["detection_tolerance_mm"])
    tp = sum(1 for d in dists_mm if d <= tol)
    fp = n_pred - tp
    fn = n_gt - tp

    metrics.update(
        {
            "ED_mm_list": dists_mm,
            "Mean_Euclidean_Distance_mm": float(np.mean(dists_mm)) if dists_mm else float("nan"),
            "Max_Euclidean_Distance_mm": float(np.max(dists_mm)) if dists_mm else float("nan"),
            "TP": tp,
            "FP": fp,
            "FN": fn,
        }
    )
    return metrics, matches, pred_centroids, gt_centroids


# ----
# Transforms and Model
# ----

def param(cfg: Config, hp: dict, key: str):
    return hp.get(key, cfg.get(key))


def make_transforms(cfg: Config, hp: dict):
    rotate90_prob = param(cfg, hp, "augment.rotate90_prob")
    flip_prob = param(cfg, hp, "augment.flip_prob")
    flip_axis = int(param(cfg, hp, "augment.flip_axis"))
    zoom_prob = param(cfg, hp, "augment.zoom_prob")
    min_zoom = param(cfg, hp, "augment.min_zoom")
    max_zoom = param(cfg, hp, "augment.max_zoom")
    gn_prob = param(cfg, hp, "augment.gauss_noise_prob")
    gn_std = param(cfg, hp, "augment.gauss_noise_std")
    gs_prob = param(cfg, hp, "augment.gauss_smooth_prob")
    gs_xmin = param(cfg, hp, "augment.gauss_sigma_x_min")
    gs_xmax = param(cfg, hp, "augment.gauss_sigma_x_max")
    ac_prob = param(cfg, hp, "augment.adjust_contrast_prob")
    gamma_min = param(cfg, hp, "augment.gamma_min")
    gamma_max = param(cfg, hp, "augment.gamma_max")

    train_tf = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image"], channel_dim=-1),
            EnsureBinaryMaskCHW(keys=["label"]),
            ScaleIntensityRanged(keys=["image"], a_min=0, a_max=255, b_min=0.0, b_max=1.0, clip=True),
            Resized(keys=["image", "label"], spatial_size=tuple(cfg.train["img_size"]), mode=["bilinear", "nearest"]),
            ToTensord(keys=["image", "label"]),
            RandRotate90d(keys=["image", "label"], prob=rotate90_prob),
            RandFlipd(keys=["image", "label"], prob=flip_prob, spatial_axis=flip_axis),
            RandZoomd(keys=["image", "label"], prob=zoom_prob, min_zoom=min_zoom, max_zoom=max_zoom, mode=["bilinear", "nearest"]),
            RandGaussianNoised(keys=["image"], prob=gn_prob, mean=0.0, std=gn_std),
            RandGaussianSmoothd(keys=["image"], prob=gs_prob, sigma_x=(gs_xmin, gs_xmax)),
            RandAdjustContrastd(keys=["image"], prob=ac_prob, gamma=(gamma_min, gamma_max)),
        ]
    )
    val_tf = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image"], channel_dim=-1),
            EnsureBinaryMaskCHW(keys=["label"]),
            ScaleIntensityRanged(keys=["image"], a_min=0, a_max=255, b_min=0.0, b_max=1.0, clip=True),
            Resized(keys=["image", "label"], spatial_size=tuple(cfg.train["img_size"]), mode=["bilinear", "nearest"]),
            ToTensord(keys=["image", "label"]),
        ]
    )
    return train_tf, val_tf


def _snap_feature_size(fs: int, mult: int = 12) -> int:
    return fs if fs % mult == 0 else int(math.ceil(fs / mult) * mult)


def sanitize_hyperparams(cfg: Config, hp: dict, logger: logging.Logger) -> dict:
    hp_sane = dict(hp)
    if "model.feature_size" in hp_sane or cfg.model.get("feature_size") is not None:
        fs_raw = int(hp_sane.get("model.feature_size", cfg.model["feature_size"]))
        fs_adj = _snap_feature_size(fs_raw, 12)
        if fs_adj != fs_raw:
            logger.info(f"[Sanitize] Adjusting model.feature_size from {fs_raw} to {fs_adj}.")
        hp_sane["model.feature_size"] = fs_adj
    dr = float(hp_sane.get("model.drop_rate", cfg.model.get("drop_rate", 0.1)))
    hp_sane["model.drop_rate"] = min(1.0, max(0.0, dr))
    bs = int(hp_sane.get("train.batch_size", cfg.train["batch_size"]))
    hp_sane["train.batch_size"] = max(1, bs)
    return hp_sane


def create_model(cfg: Config, hp: dict) -> nn.Module:
    img_size = tuple(cfg.train["img_size"])
    feature_size = int(param(cfg, hp, "model.feature_size"))
    drop_rate = float(param(cfg, hp, "model.drop_rate"))
    in_ch = int(param(cfg, hp, "model.in_channels"))
    out_ch = int(param(cfg, hp, "model.out_channels"))
    spatial_dims = int(param(cfg, hp, "model.spatial_dims"))
    use_checkpoint = bool(param(cfg, hp, "model.use_checkpoint"))

    model = SwinUNETR(
        img_size=img_size,
        in_channels=in_ch,
        out_channels=out_ch,
        feature_size=feature_size,
        spatial_dims=spatial_dims,
        drop_rate=drop_rate,
        use_checkpoint=use_checkpoint,
    ).to(cfg.device)

    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)) and getattr(m, "weight", None) is not None:
            nn.init.kaiming_normal_(m.weight)
    return model


def build_losses(cfg: Config, device: torch.device, hp: dict):
    class_weight = float(param(cfg, hp, "train.class_weight"))
    focal_gamma = float(cfg.loss.get("focal_gamma", 3.0))
    tversky_alpha = float(cfg.loss.get("tversky_alpha", 0.6))
    tversky_beta = float(cfg.loss.get("tversky_beta", 0.4))
    comb_w = cfg.loss.get("combined_weights", [0.1, 0.7, 0.2])
    w_dice, w_focal, w_tversky = float(comb_w[0]), float(comb_w[1]), float(comb_w[2])

    dice = DiceLoss(sigmoid=True)
    focal = FocalLoss(gamma=focal_gamma, weight=torch.tensor([class_weight], dtype=torch.float32, device=device))
    tversky = TverskyLoss(sigmoid=True, alpha=tversky_alpha, beta=tversky_beta)

    def combined(outputs, labels):
        return w_dice * dice(outputs, labels) + w_focal * focal(outputs, labels) + w_tversky * tversky(outputs, labels)

    return dice, focal, tversky, combined


# ----
# I/O helpers
# ----

def create_patient_dirs(base_dir: str, patient_id: str) -> str:
    directory = os.path.join(base_dir, patient_id)
    os.makedirs(directory, exist_ok=True)
    return directory


def save_json(path: str, data: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def save_fold_hparams_json(cfg: Config, fold_idx: int, hp: dict, extra: Optional[dict] = None):
    d = dict(
        timestamp=cfg.timestamp,
        fold=fold_idx,
        model="SwinUNETR",
        in_channels=param(cfg, hp, "model.in_channels"),
        out_channels=param(cfg, hp, "model.out_channels"),
        img_size=list(cfg.train["img_size"]),
        batch_size=param(cfg, hp, "train.batch_size"),
        accumulation_steps=param(cfg, hp, "train.accumulation_steps"),
        max_epochs=param(cfg, hp, "train.max_epochs"),
        patience=param(cfg, hp, "train.patience"),
        lr=param(cfg, hp, "train.learning_rate"),
        class_weight=param(cfg, hp, "train.class_weight"),
        feature_size=param(cfg, hp, "model.feature_size"),
        drop_rate=param(cfg, hp, "model.drop_rate"),
        pred_threshold=param(cfg, hp, "metrics.pred_threshold"),
        min_component_size=cfg.metrics["min_component_size"],
        pixel_to_mm=cfg.metrics["pixel_to_mm"],
        detection_tolerance_mm=cfg.metrics["detection_tolerance_mm"],
        k_folds=cfg.search["cv_n_splits"],
        cv_repeats=cfg.search["cv_n_repeats"],
        random_state=cfg.cv["random_state"],
        test_size=cfg.cv["test_size"],
        evaluation_fold=cfg.cv["evaluation_fold"],
        augmentations=True,
        description=cfg.description,
    )
    if extra:
        d.update(extra)
    path = os.path.join(cfg.folds_json_dir, f"fold_{fold_idx}_hyperparams.json")
    save_json(path, d)
    return path, d


def update_aggregate_json(cfg: Config, fold_entries: list, cv_summary: dict):
    out = {"timestamp": cfg.timestamp, "folds": fold_entries, "cv_summary": cv_summary}
    path = os.path.join(cfg.base_output_dir, "aggregate_summary.json")
    save_json(path, out)
    return path


# ----
# Training primitives
# ----

def build_loaders_from_ids(cfg: Config, hp: dict, patient_data: dict, tr_ids: List[str], va_ids: List[str]):
    train_tf, val_tf = make_transforms(cfg, hp)
    tr_items, va_items = [], []
    for pid in tr_ids:
        tr_items.extend(patient_data[pid])
    for pid in va_ids:
        va_items.extend(patient_data[pid])
    if len(va_items) == 0:
        va_items = tr_items[:1]
    weights = [1.0 for _ in tr_items]
    sampler = WeightedRandomSampler(weights=weights, num_samples=len(tr_items), replacement=True)
    train_loader = DataLoader(Dataset(tr_items, transform=train_tf),
                              batch_size=int(param(cfg, hp, "train.batch_size")),
                              sampler=sampler,
                              num_workers=int(cfg.train.get("num_workers", 0)),
                              pin_memory=(cfg.device.type == "cuda"))
    val_loader = DataLoader(Dataset(va_items, transform=val_tf),
                            batch_size=1,
                            shuffle=False,
                            num_workers=int(cfg.train.get("num_workers", 0)),
                            pin_memory=(cfg.device.type == "cuda"))
    return train_loader, val_loader


def evaluate_validation_epoch(cfg: Config, hp: dict, model: nn.Module, val_loader, losses):
    dice_loss, focal_loss, tversky_loss, combined_loss = losses
    model.eval()
    amp_enabled = bool(cfg.train.get("use_amp", True)) and (cfg.device.type == "cuda")

    val_dice_vals, val_focal_vals, val_tversky_vals, val_comb_vals = [], [], [], []
    all_ed_mm: List[float] = []
    TP_total = FP_total = FN_total = 0

    with torch.no_grad():
        for batch in val_loader:
            vimg = batch["image"].to(cfg.device, non_blocking=True)
            vmsk = batch["label"].to(cfg.device, non_blocking=True)
            file_id, H_orig, W_orig = _get_img_info_from_batch(batch)

            with torch.amp.autocast("cuda", enabled=amp_enabled):
                logits = model(vimg)
                probs = torch.sigmoid(logits)
                thr = float(param(cfg, hp, "metrics.pred_threshold"))
                bin_pred = (probs > thr).float()

            val_dice_vals.append(float(dice_loss(logits, vmsk).item()))
            val_focal_vals.append(float(focal_loss(logits, vmsk).item()))
            val_tversky_vals.append(float(tversky_loss(logits, vmsk).item()))
            val_comb_vals.append(float(combined_loss(logits, vmsk).item()))

            metrics_img, _, _, _ = compute_euclidean_mm_and_matches(
                cfg, bin_pred[0], vmsk[0], file_id, (H_orig, W_orig), tuple(cfg.train["img_size"])
            )
            all_ed_mm.extend(metrics_img["ED_mm_list"])
            TP_total += metrics_img["TP"]
            FP_total += metrics_img["FP"]
            FN_total += metrics_img["FN"]

    mean_dice = float(np.mean(val_dice_vals)) if val_dice_vals else float("nan")
    mean_focal = float(np.mean(val_focal_vals)) if val_focal_vals else float("nan")
    mean_tversky = float(np.mean(val_tversky_vals)) if val_tversky_vals else float("nan")
    mean_comb = float(np.mean(val_comb_vals)) if val_comb_vals else float("nan")

    if all_ed_mm:
        mean_ed_mm = float(np.mean(all_ed_mm))
        max_ed_mm = float(np.max(all_ed_mm))
        std_ed_mm = float(np.std(all_ed_mm))
    else:
        mean_ed_mm = float("nan")
        max_ed_mm = float("nan")
        std_ed_mm = float("nan")

    precision = TP_total / (TP_total + FP_total) if (TP_total + FP_total) > 0 else 0.0
    recall = TP_total / (TP_total + FN_total) if (TP_total + FN_total) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "val_dice": mean_dice,
        "val_focal": mean_focal,
        "val_tversky": mean_tversky,
        "val_combined": mean_comb,
        "mean_euclidean_mm": mean_ed_mm,
        "max_euclidean_mm": max_ed_mm,
        "std_euclidean_mm": std_ed_mm,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _get_original_hw_from_file(file_path: str) -> Tuple[Optional[int], Optional[int]]:
    try:
        with Image.open(file_path) as im:
            w, h = im.size
            return h, w
    except Exception:
        return None, None


def _get_img_info_from_batch(batch) -> Tuple[str, Optional[int], Optional[int]]:
    if "image_path" in batch:
        img_path = batch["image_path"]
        if isinstance(img_path, (list, tuple)):
            img_path = img_path[0]
        file_id = os.path.basename(img_path)
        H_orig, W_orig = _get_original_hw_from_file(img_path)
        return file_id, H_orig, W_orig

    if "image_meta_dict" in batch:
        meta = batch["image_meta_dict"]
        fn = None
        if isinstance(meta, (list, tuple)) and len(meta) > 0 and isinstance(meta[0], dict):
            fn = meta[0].get("filename_or_obj", [None])[0]
        elif isinstance(meta, dict):
            fn = meta.get("filename_or_obj", [None])[0]
        if fn is not None:
            file_id = os.path.basename(fn)
            H_orig, W_orig = _get_original_hw_from_file(fn)
            return file_id, H_orig, W_orig

    return "unknown.png", None, None


def train_one_fold(cfg: Config, hp: dict, train_loader, val_loader, fold_idx: int, fold_dir: str, logger: logging.Logger, max_epochs_override: Optional[int] = None, epoch_csv_outdir: Optional[str] = None):
    device = cfg.device
    model = create_model(cfg, hp)

    lr = float(param(cfg, hp, "train.learning_rate"))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=float(cfg.train.get("plateau_reduce_factor", 0.5)), patience=int(cfg.train.get("plateau_patience_epochs", 3)),
    )
    dice_loss, focal_loss, tversky_loss, combined_loss = build_losses(cfg, device, hp)

    amp_enabled = bool(cfg.train.get("use_amp", True)) and (device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    train_bsz = int(param(cfg, hp, "train.batch_size"))
    cfg_mbs = cfg.train.get("micro_batch_size", None)
    base_micro_bsz = max(1, min(train_bsz, int(cfg_mbs))) if cfg_mbs is not None else train_bsz
    auto_mbs = bool(cfg.train.get("auto_microbatch_on_oom", True))
    max_halvings = int(cfg.train.get("max_auto_microbatch_halvings", 3))

    best_metric = float("inf")
    best_epoch = -1
    patience_ctr = 0
    patience = int(param(cfg, hp, "train.patience"))
    min_delta = float(cfg.train.get("early_stop_min_delta_mm", 0.0))
    max_epochs = int(param(cfg, hp, "train.max_epochs")) if max_epochs_override is None else int(max_epochs_override)
    acc_steps = int(param(cfg, hp, "train.accumulation_steps"))
    best_model_path = os.path.join(fold_dir, "best_model.pth")
    per_epoch_rows = []
    step = 0

    os.makedirs(fold_dir, exist_ok=True)
    if epoch_csv_outdir is not None:
        os.makedirs(epoch_csv_outdir, exist_ok=True)

    for epoch in range(1, max_epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        epoch_loss = 0.0

        for batch in train_loader:
            inputs = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            B = inputs.shape[0]

            local_mbs = min(base_micro_bsz, B)
            num_chunks = math.ceil(B / local_mbs)

            halvings_done = 0
            did_work = False
            while True:
                try:
                    chunk_loss_accum = 0.0
                    for start in range(0, B, local_mbs):
                        end = min(start + local_mbs, B)
                        in_i = inputs[start:end]
                        lb_i = labels[start:end]
                        with torch.amp.autocast("cuda", enabled=amp_enabled):
                            logits = model(in_i)
                            loss_i = (combined_loss(logits, lb_i) / float(num_chunks) / float(acc_steps))
                        scaler.scale(loss_i).backward()
                        chunk_loss_accum += float(loss_i.item())
                    epoch_loss += chunk_loss_accum * acc_steps
                    did_work = True
                    break
                except RuntimeError as e:
                    if (device.type == "cuda") and ("out of memory" in str(e).lower()):
                        logger.warning(f"[CUDA OOM] epoch={epoch} step={step+1} B={B}, local_mbs={local_mbs}. "
                                       f"{'Retrying' if auto_mbs and local_mbs > 1 and halvings_done < max_halvings else 'Skipping batch'}.")
                        try:
                            del in_i, lb_i, logits
                        except Exception:
                            pass
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        gc.collect()
                        time.sleep(0.2)
                        if auto_mbs and local_mbs > 1 and halvings_done < max_halvings:
                            local_mbs = max(1, local_mbs // 2)
                            num_chunks = math.ceil(B / local_mbs)
                            halvings_done += 1
                            continue
                        else:
                            try:
                                del inputs, labels
                            except Exception:
                                pass
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            gc.collect()
                            optimizer.zero_grad(set_to_none=True)
                            did_work = False
                            break
                    else:
                        raise
            if not did_work:
                continue

            step += 1
            if step % acc_steps == 0:
                try:
                    scaler.unscale_(optimizer)
                except Exception:
                    pass
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

        # flush partial accumulation at epoch end
        if step % acc_steps != 0:
            try:
                scaler.unscale_(optimizer)
            except Exception:
                pass
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        avg_train_loss = epoch_loss / max(step, 1)

        # Validation
        val_metrics = evaluate_validation_epoch(cfg, hp, model, val_loader, (dice_loss, focal_loss, tversky_loss, combined_loss))
        mean_ed_mm = val_metrics["mean_euclidean_mm"]
        scheduler.step(mean_ed_mm if math.isfinite(mean_ed_mm) else 1e6)

        improved = False
        if epoch == 1 and best_epoch == -1:
            torch.save(model.state_dict(), best_model_path)
            best_metric = mean_ed_mm if math.isfinite(mean_ed_mm) else 1e6
            best_epoch = 1
            improved = True
        elif math.isfinite(mean_ed_mm) and (((best_metric - mean_ed_mm) >= min_delta) and (mean_ed_mm < best_metric)):
            best_metric = mean_ed_mm
            best_epoch = epoch
            improved = True
            torch.save(model.state_dict(), best_model_path)

        patience_ctr = 0 if improved else (patience_ctr + 1)

        per_epoch_rows.append({"epoch": epoch, "train_loss": avg_train_loss, **val_metrics, "is_best_epoch": False})

        cfg_best = best_metric if math.isfinite(best_metric) else float("nan")
        cfg_mean = mean_ed_mm if math.isfinite(mean_ed_mm) else float("nan")
        logger.info(f"Fold {fold_idx} | Epoch {epoch}/{max_epochs} | train {avg_train_loss:.4f} | "
                    f"val_comb {val_metrics['val_combined']:.4f} | meanED {cfg_mean:.3f} mm | best@{best_epoch}:{cfg_best:.3f} mm")

        if patience_ctr >= patience:
            logger.info(f"Fold {fold_idx}: Early stopping at epoch {epoch}.")
            break

    if best_epoch > 0 and 1 <= best_epoch <= len(per_epoch_rows):
        per_epoch_rows[best_epoch - 1]["is_best_epoch"] = True

    epoch_outdir = epoch_csv_outdir if epoch_csv_outdir is not None else cfg.epoch_csv_dir
    os.makedirs(epoch_outdir, exist_ok=True)
    fold_epoch_csv = os.path.join(epoch_outdir, f"fold{fold_idx}_per_epoch.csv")
    pd.DataFrame(per_epoch_rows).to_csv(fold_epoch_csv, index=False)

    logger.info(f"Fold {fold_idx} completed. Best mean ED(mm): "
                f"{best_metric if math.isfinite(best_metric) else float('nan'):.3f} at epoch {best_epoch}.",
                extra={"milestone": True})
    best_val_metrics = per_epoch_rows[best_epoch - 1].copy() if best_epoch > 0 else None

    try:
        del model
    except Exception:
        pass
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    return best_metric, best_epoch, best_model_path, fold_epoch_csv, best_val_metrics


# ----
# Search helpers (Optuna only)
# ----

def _write_per_trial_summaries(strat_dir: str, results: list):
    rows, agg = [], []
    for rec in results:
        tnum = rec.get("trial")
        hp = rec.get("params", {})
        fold_details = rec.get("fold_details", [])
        for fi, m in enumerate(fold_details, start=1):
            rows.append(
                {
                    "trial": tnum,
                    "fold": fi,
                    "best_epoch": m.get("epoch"),
                    "val_mean_euclidean_mm": m.get("mean_euclidean_mm"),
                    "val_max_euclidean_mm": m.get("max_euclidean_mm"),
                    "val_std_euclidean_mm": m.get("std_euclidean_mm"),
                    "precision": m.get("precision"),
                    "recall": m.get("recall"),
                    "f1": m.get("f1"),
                    **{f"hp::{k}": v for k, v in hp.items()},
                }
            )
        ms = rec.get("mean_scores", {})
        agg.append({"trial": tnum, "cv_mean_f1": ms.get("f1"), "cv_mean_ed_mm": ms.get("mean_ed_mm"), **{f"hp::{k}": v for k, v in hp.items()}})
    if rows:
        pd.DataFrame(rows).to_csv(os.path.join(strat_dir, "per_trial_fold_summary.csv"), index=False)
    if agg:
        pd.DataFrame(agg).to_csv(os.path.join(strat_dir, "per_trial_summary.csv"), index=False)


def _suggest_from_space(trial, space: dict) -> dict:
    hp = {}
    for key, spec in space.items():
        t = spec.get("type", "float")
        if t == "float":
            low = float(spec["low"]); high = float(spec["high"])
            log = bool(spec.get("log", False))
            step = spec.get("step", None)
            if step is not None:
                hp[key] = trial.suggest_float(key, low, high, step=float(step), log=log)
            else:
                hp[key] = trial.suggest_float(key, low, high, log=log)
        elif t == "int":
            low = int(spec["low"]); high = int(spec["high"]); step = int(spec.get("step", 1))
            hp[key] = trial.suggest_int(key, low, high, step=step)
        elif t == "categorical":
            choices = spec["choices"]
            hp[key] = trial.suggest_categorical(key, choices)
        else:
            raise ValueError(f"Unknown type for {key}: {t}")
    return hp


def evaluate_hp_via_cv(cfg: Config, logger: logging.Logger, patient_data: dict, train_ids: List[str], strat_labels: dict, hp_raw: dict, base_dir: str, n_splits: int, n_repeats: int, cv_max_epochs: int, prefix: str):
    hp = sanitize_hyperparams(cfg, hp_raw, logger)
    os.makedirs(base_dir, exist_ok=True)

    y = np.array([strat_labels[pid] for pid in train_ids])
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=cfg.cv["random_state"])

    fold_scores = []
    fold_metrics_collect = []

    for rep_fold_idx, (tr_idx, va_idx) in enumerate(rskf.split(train_ids, y), start=1):
        fold_dir = os.path.join(base_dir, f"{prefix}_fold_{rep_fold_idx}")
        os.makedirs(fold_dir, exist_ok=True)

        tr_pids = [train_ids[i] for i in tr_idx]
        va_pids = [train_ids[i] for i in va_idx]

        train_loader, val_loader = build_loaders_from_ids(cfg, hp, patient_data, tr_pids, va_pids)
        _, _, _, _, best_val_metrics = train_one_fold(
            cfg, hp, train_loader, val_loader, fold_idx=rep_fold_idx, fold_dir=fold_dir, logger=logger, max_epochs_override=cv_max_epochs, epoch_csv_outdir=os.path.join(cfg.epoch_csv_dir, prefix)
        )

        f1 = best_val_metrics["f1"] if best_val_metrics and ("f1" in best_val_metrics) else 0.0
        mean_ed = best_val_metrics["mean_euclidean_mm"] if best_val_metrics else float("inf")

        fold_scores.append((f1, mean_ed))
        fold_metrics_collect.append(best_val_metrics or {})

        if cfg.device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    mean_f1 = float(np.mean([s[0] for s in fold_scores])) if fold_scores else 0.0
    mean_ed = float(np.mean([s[1] for s in fold_scores])) if fold_scores else float("inf")

    result = {"params": hp, "mean_scores": {"f1": mean_f1, "mean_ed_mm": mean_ed}, "fold_details": fold_metrics_collect}
    return result


def bayes_opt_search(cfg: Config, logger, patient_data, train_ids, strat_labels):
    try:
        import optuna
    except Exception as e:
        raise RuntimeError("Optuna is required for bayes_opt strategy. Install with: pip install optuna") from e

    params = cfg.search.get("bayes_opt", {}).get("param_space", {})
    if not params:
        raise ValueError("Bayesian optimization requires non-empty search.bayes_opt.param_space in config.")

    n_trials = int(cfg.search.get("bayes_opt", {}).get("n_trials", 15))
    seed = int(cfg.search.get("bayes_opt", {}).get("seed", 42))
    n_splits = int(cfg.search["cv_n_splits"])
    n_repeats = int(cfg.search["cv_n_repeats"])
    cv_max_epochs = int(cfg.search["cv_max_epochs"])

    strat_dir = os.path.join(cfg.search_dir, "bayes_opt")
    os.makedirs(strat_dir, exist_ok=True)
    save_json(os.path.join(strat_dir, "param_space.json"), params)

    results = []
    trials_path = os.path.join(strat_dir, "trials.jsonl")
    with open(trials_path, "w"): pass

    def objective(trial):
        hp = _suggest_from_space(trial, params)
        if "model.feature_size" in hp:
            hp["model.feature_size"] = _snap_feature_size(int(hp["model.feature_size"]), 12)

        base_dir = os.path.join(strat_dir, f"trial_{trial.number + 1}")
        try:
            res = evaluate_hp_via_cv(cfg, logger, patient_data, train_ids, strat_labels, hp, base_dir, n_splits, n_repeats, cv_max_epochs, prefix=f"trial_{trial.number + 1}")
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.warning(f"[Optuna] Trial {trial.number} OOM. hp={hp}. Returning poor score.")
                try:
                    trial.set_user_attr("oom", True)
                except Exception:
                    pass
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                time.sleep(0.2)
                return -1e9
            else:
                raise

        results.append({"trial": trial.number + 1, **res})
        with open(trials_path, "a") as f_out:
            f_out.write(json.dumps(results[-1]) + "\n")

        try:
            trial.set_user_attr("mean_f1", res["mean_scores"]["f1"])
            trial.set_user_attr("mean_ed_mm", res["mean_scores"]["mean_ed_mm"])
        except Exception:
            pass
        return res["mean_scores"]["f1"]

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials)

    collected = []
    for t in study.trials:
        hp = {}
        for k in params.keys():
            hp[k] = t.params.get(k)
        if "model.feature_size" in hp and hp["model.feature_size"] is not None:
            hp["model.feature_size"] = _snap_feature_size(int(hp["model.feature_size"]), 12)
        collected.append(
            {
                "trial": t.number + 1,
                "params": hp,
                "mean_scores": {"f1": t.user_attrs.get("mean_f1", 0.0), "mean_ed_mm": t.user_attrs.get("mean_ed_mm", float("inf"))},
                "optuna_value": t.value,
            }
        )

    save_json(os.path.join(strat_dir, "summary.json"), {"trials": collected})
    best = max(collected, key=lambda r: (r["mean_scores"].get("f1", 0.0), -r["mean_scores"].get("mean_ed_mm", float("inf")))) if collected else None
    save_json(os.path.join(strat_dir, "best.json"), best or {})
    _write_per_trial_summaries(strat_dir, results)
    return best, collected


# ----
# Uncertainty helpers
# ----

def _entropy_from_prob_map(p: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    p = np.clip(p, eps, 1.0 - eps)
    return -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))


def _set_dropout_train_only(model: nn.Module, enable: bool) -> List[nn.Module]:
    affected = []
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            affected.append(m)
            if enable: m.train()
            else: m.eval()
    return affected


def _mc_dropout_prob_stack(cfg: Config, model: nn.Module, img_tensor: torch.Tensor, num_samples: int, seed: Optional[int], amp_enabled: bool) -> np.ndarray:
    device = cfg.device
    model.eval()
    _ = _set_dropout_train_only(model, True)
    probs_list = []
    with torch.no_grad():
        for i in range(num_samples):
            if seed is not None:
                torch.manual_seed(int(seed) + i)
                np.random.seed(int(seed) + i)
                random.seed(int(seed) + i)
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                logits_i = model(img_tensor.to(device, non_blocking=True))
                pi = torch.sigmoid(logits_i)[0, 0].detach().cpu().numpy()
                probs_list.append(pi)
    _ = _set_dropout_train_only(model, False)
    if len(probs_list) == 0:
        return np.zeros(tuple(cfg.train["img_size"]), dtype=np.float32)
    return np.stack(probs_list, axis=0)


def _pool_windowed_values(arr: np.ndarray, points_xy: List[Tuple[int, int]], window: int, pool_type: str) -> List[float]:
    H, W = arr.shape
    r = max(0, window // 2)
    vals = []
    for (x, y) in points_xy:
        x0 = max(0, x - r); x1 = min(W, x + r + 1)
        y0 = max(0, y - r); y1 = min(H, y + r + 1)
        patch = arr[y0:y1, x0:x1]
        if patch.size == 0:
            vals.append(float("nan"))
        else:
            vals.append(float(np.nanmax(patch)) if pool_type == "max" else float(np.nanmean(patch)))
    return vals


def _minmax_norm_list(vals: List[float]) -> List[float]:
    arr = np.array(vals, dtype=float)
    mask = np.isfinite(arr)
    if not np.any(mask):
        return [float("nan")] * len(vals)
    vmin = float(np.min(arr[mask])); vmax = float(np.max(arr[mask]))
    if vmax <= vmin + 1e-12:
        return [0.0 if m else float("nan") for m in mask]
    out = (arr - vmin) / (vmax - vmin)
    out[~mask] = np.nan
    return [float(v) for v in out.tolist()]


def _minmax_norm_map(arr: np.ndarray) -> np.ndarray:
    a = arr.astype(np.float32)
    finite = np.isfinite(a)
    if not np.any(finite):
        return np.zeros_like(a, dtype=np.float32)
    vmin = float(np.min(a[finite])); vmax = float(np.max(a[finite]))
    if vmax <= vmin + 1e-12:
        out = np.zeros_like(a, dtype=np.float32)
        out[~finite] = 0.0
        return out
    out = (a - vmin) / (vmax - vmin)
    out[~finite] = 0.0
    return out


def _apply_colormap(arr01: np.ndarray, cmap_name: str = "magma") -> np.ndarray:
    arr01 = np.clip(arr01.astype(np.float32), 0.0, 1.0)
    cmap = mpl_cm.get_cmap(cmap_name)
    rgba = cmap(arr01)  # HxWx4, float in [0,1]
    rgb = (rgba[..., :3] * 255.0).astype(np.uint8)
    return rgb


def _save_unc_map_png(path: str, arr01: np.ndarray) -> None:
    # Retained for backward compatibility (not used by new path)
    arr = np.clip(arr01, 0.0, 1.0)
    img = (arr * 255.0).astype(np.uint8)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(img).save(path)


def _save_unc_map_colormap(path: str, arr01: np.ndarray, cmap_name: str) -> None:
    rgb = _apply_colormap(arr01, cmap_name=cmap_name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(rgb).save(path)


# ----
# Evaluation and visualization (with uncertainty)
# ----

def centroids_to_coords(centroids, slice_num: int, max_z: int):
    coords = []
    for (x, y) in centroids:
        z = max_z - slice_num
        coords.append((x, y, z))
    return coords


def save_needle_coordinates(cfg: Config, patient_id: str, slices_bundle: List[dict], max_z: int, fold_or_tag: str, mm_per_pixel: float):
    rows = max_z + 1
    max_needles = 0
    for s in slices_bundle:
        n_pred = len(s["pred_centroids"])
        n_gt = len(s["gt_centroids"])
        max_needles = max(max_needles, max(n_pred, n_gt))

    match_df: Dict[str, List[float]] = {}
    for k in range(max_needles):
        name = f"Needle_{k+1}"
        for col in [f"{name}_X", f"{name}_Y", f"{name}_Z", f"{name}_gt_X", f"{name}_gt_Y", f"{name}_gt_Z", f"{name}_ED_mm",
                    f"{name}_Prob", f"{name}_PE", f"{name}_EPI", f"{name}_Unc"]:
            match_df[col] = [np.nan] * rows

    for s in slices_bundle:
        z_row = max_z - s["slice_num"]
        pred_c = s["pred_centroids"]
        pred_p = s.get("pred_probs", [])
        pred_pe = s.get("pred_pe", [])
        pred_epi = s.get("pred_epi", [])
        pred_unc = s.get("pred_unc", [])
        gt_c = s["gt_centroids"]
        matches = s["matches"]

        used_p = {pi for (pi, _) in matches}
        used_g = {gi for (_, gi) in matches}
        pairs_list = [(pi, gi, True) for (pi, gi) in matches]
        pairs_list += [(pi, None, False) for pi in range(len(pred_c)) if pi not in used_p]
        pairs_list += [(None, gi, False) for gi in range(len(gt_c)) if gi not in used_g]

        for idx, (pi, gi, is_matched) in enumerate(pairs_list):
            if idx >= max_needles:
                break
            name = f"Needle_{idx+1}"
            if pi is not None:
                x, y = pred_c[pi]
                match_df[f"{name}_X"][z_row] = x
                match_df[f"{name}_Y"][z_row] = y
                match_df[f"{name}_Z"][z_row] = z_row
                if 0 <= pi < len(pred_p):
                    match_df[f"{name}_Prob"][z_row] = float(pred_p[pi])
                if 0 <= pi < len(pred_pe):
                    match_df[f"{name}_PE"][z_row] = float(pred_pe[pi])
                if 0 <= pi < len(pred_epi):
                    match_df[f"{name}_EPI"][z_row] = float(pred_epi[pi])
                if 0 <= pi < len(pred_unc):
                    match_df[f"{name}_Unc"][z_row] = float(pred_unc[pi])
            if gi is not None:
                gx, gy = gt_c[gi]
                match_df[f"{name}_gt_X"][z_row] = gx
                match_df[f"{name}_gt_Y"][z_row] = gy
                match_df[f"{name}_gt_Z"][z_row] = z_row
            if is_matched and pi is not None and gi is not None:
                x, y = pred_c[pi]; gx, gy = gt_c[gi]
                d_mm = math.hypot(x - gx, y - gy) * mm_per_pixel
                match_df[f"{name}_ED_mm"][z_row] = d_mm

    fold_csv_dir = os.path.join(cfg.coords_dir, f"fold_{fold_or_tag}")
    os.makedirs(fold_csv_dir, exist_ok=True)
    match_csv_path = os.path.join(fold_csv_dir, f"{patient_id}_matching_pairs.csv")
    pd.DataFrame(match_df).to_csv(match_csv_path, index=False)

    pred_df, gt_df = {}, {}
    for k in range(max_needles):
        name = f"Needle_{k+1}"
        pred_df[f"{name}_X"] = match_df[f"{name}_X"]
        pred_df[f"{name}_Y"] = match_df[f"{name}_Y"]
        pred_df[f"{name}_Z"] = match_df[f"{name}_Z"]
        pred_df[f"{name}_Prob"] = match_df[f"{name}_Prob"]
        pred_df[f"{name}_PE"] = match_df[f"{name}_PE"]
        pred_df[f"{name}_EPI"] = match_df[f"{name}_EPI"]
        pred_df[f"{name}_Unc"] = match_df[f"{name}_Unc"]
        gt_df[f"{name}_X"] = match_df[f"{name}_gt_X"]
        gt_df[f"{name}_Y"] = match_df[f"{name}_gt_Y"]
        gt_df[f"{name}_Z"] = match_df[f"{name}_gt_Z"]

    pred_csv_path = os.path.join(fold_csv_dir, f"{patient_id}_needle_coordinates.csv")
    gt_csv_path = os.path.join(fold_csv_dir, f"{patient_id}_gt_needle_coordinates.csv")
    pd.DataFrame(pred_df).to_csv(pred_csv_path, index=False)
    pd.DataFrame(gt_df).to_csv(gt_csv_path, index=False)
    return pred_csv_path, gt_csv_path, match_csv_path


def _try_load_font(font_path: Optional[str], font_size: int):
    if font_path is None:
        return ImageFont.load_default()
    try:
        return ImageFont.truetype(font_path, font_size)
    except Exception:
        return ImageFont.load_default()


def _extract_pred_probs(cfg: Config, bin_pred_01: np.ndarray, prob_map: np.ndarray, pred_centroids: List[Tuple[int, int]]):
    method = str(cfg.metrics.get("prob_value_method", "centroid")).lower().strip()
    H, W = prob_map.shape
    if method == "centroid":
        probs = []
        for (x, y) in pred_centroids:
            if 0 <= x < W and 0 <= y < H:
                probs.append(float(prob_map[y, x]))
            else:
                probs.append(float("nan"))
        return probs
    elif method == "component_mean":
        labeled, n = cc_label(bin_pred_01.astype(np.uint8))
        comp_means: Dict[int, float] = {}
        for i in range(1, n + 1):
            mask_i = labeled == i
            if mask_i.any():
                comp_means[i] = float(prob_map[mask_i].mean())
        probs = []
        for (x, y) in pred_centroids:
            if 0 <= x < W and 0 <= y < H:
                comp_id = int(labeled[y, x])
                probs.append(comp_means.get(comp_id, float("nan")))
            else:
                probs.append(float("nan"))
        return probs
    else:
        return [float("nan")] * len(pred_centroids)


def _rotate_if_needed(np_img: np.ndarray, rotate_cw_90: bool) -> np.ndarray:
    return np.rot90(np_img, k=-1) if rotate_cw_90 else np_img


def _rotate_points_cw90(points_xy: List[Tuple[int, int]], H: int, W: int) -> List[Tuple[int, int]]:
    out = []
    for (x, y) in points_xy:
        x2 = H - 1 - y; y2 = x
        out.append((int(x2), int(y2)))
    return out


def _scale_points(points_xy: List[Tuple[int, int]], from_hw: Tuple[int, int], to_hw: Tuple[int, int]) -> List[Tuple[int, int]]:
    fy, fx = from_hw[0], from_hw[1]; ty, tx = to_hw[0], to_hw[1]
    sx = tx / float(fx); sy = ty / float(fy)
    return [(int(round(x * sx)), int(round(y * sy))) for (x, y) in points_xy]


def _clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def _apply_overlay_only_orientation(points_xy: List[Tuple[int, int]], H: int, W: int, mode: str) -> List[Tuple[int, int]]:
    mode = (mode or "none").lower().strip()
    if mode == "none":
        return points_xy
    if mode == "rot90_ccw":
        transformed = []
        for (x, y) in points_xy:
            x1 = W - 1 - x; y1 = y
            x2 = y1; y2 = W - 1 - x1
            x3 = int(round(x2 * (W / float(H))))
            y3 = int(round(y2 * (H / float(W))))
            x3 = _clamp(x3, 0, W - 1); y3 = _clamp(y3, 0, H - 1)
            transformed.append((x3, y3))
        return transformed
    return points_xy


def _draw_pred_points_on_axes(ax, points_xy: List[Tuple[int, int]], radius_px: int, face_rgb: Tuple[float, float, float], edge_rgb: Tuple[float, float, float], lw: float, show_probs: bool = False, probs: Optional[List[float]] = None, digits: int = 2, text_rgb: Tuple[float, float, float] = (1, 1, 0)):
    for idx, (x, y) in enumerate(points_xy):
        circ = Circle((x, y), radius_px, facecolor=face_rgb, edgecolor=edge_rgb, linewidth=lw, alpha=0.9)
        ax.add_patch(circ)
        if show_probs and probs is not None and idx < len(probs) and probs[idx] == probs[idx]:
            txt = str(round(float(probs[idx]), digits))
            ax.text(x + radius_px + 2, y - radius_px - 2, txt, color=text_rgb, fontsize=9, ha="left", va="top")


def _show_image(ax, img: np.ndarray, title: str, title_fs: int):
    H, W = img.shape[:2]
    ax.imshow(img, origin="upper", extent=[0, W, H, 0], cmap="gray" if img.ndim == 2 else None, aspect="equal")
    ax.set_xlim(0, W); ax.set_ylim(H, 0)
    ax.set_title(title, fontsize=title_fs)
    ax.axis("off")
    return H, W


def _show_uncert_map(ax, arr01: np.ndarray, title: str, title_fs: int, cmap_name: str, show_colorbar: bool):
    H, W = arr01.shape
    im = ax.imshow(arr01, origin="upper", extent=[0, W, H, 0], cmap=cmap_name, vmin=0.0, vmax=1.0, aspect="equal")
    ax.set_xlim(0, W); ax.set_ylim(H, 0)
    ax.set_title(title, fontsize=title_fs)
    ax.axis("off")
    if show_colorbar:
        # Shrink colorbar to be compact per panel
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.05)
        plt.colorbar(im, cax=cax)


def _save_slice_collage_matplotlib(cfg: Config, raw_path: str, mask_path: str, pred_centroids: List[Tuple[int, int]], gt_centroids: List[Tuple[int, int]], pred_probs: List[float], resized_hw: Tuple[int, int], out_path: str, title_input: str, title_gt: str, title_pred_on_input: str, title_pred_on_gt: str):
    raw = np.array(Image.open(raw_path).convert("RGB"))
    gt = np.array(Image.open(mask_path).convert("L"))
    gt_rgb = np.stack([gt, gt, gt], axis=-1)

    rH, rW = int(resized_hw[0]), int(resized_hw[1])
    H_raw, W_raw = raw.shape[0], raw.shape[1]
    H_gt, W_gt = gt_rgb.shape[0], gt_rgb.shape[1]

    pred_on_raw = _scale_points(pred_centroids, (rH, rW), (H_raw, W_raw))
    pred_on_gt = _scale_points(pred_centroids, (rH, rW), (H_gt, W_gt))

    rotate = bool(cfg.viz.get("rotate_cw_90", False))
    if rotate:
        raw = _rotate_if_needed(raw, True)
        gt_rgb = _rotate_if_needed(gt_rgb, True)
        pred_on_raw = _rotate_points_cw90(pred_on_raw, H_raw, W_raw)
        pred_on_gt = _rotate_points_cw90(pred_on_gt, H_gt, W_gt)
        H_raw, W_raw = raw.shape[0], raw.shape[1]
        H_gt, W_gt = gt_rgb.shape[0], gt_rgb.shape[1]

    overlay_mode = str(cfg.viz.get("overlay_orientation", "none")).lower().strip()
    if overlay_mode != "none":
        pred_on_raw = _apply_overlay_only_orientation(pred_on_raw, H_raw, W_raw, overlay_mode)
        pred_on_gt = _apply_overlay_only_orientation(pred_on_gt, H_gt, W_gt, overlay_mode)

    face_rgb = tuple(float(v) for v in cfg.viz.get("pred_point_color", [0, 1, 0]))
    edge_rgb = tuple(float(v) for v in cfg.viz.get("pred_point_edgecolor", [1, 1, 0]))
    radius_base = int(cfg.viz.get("pred_point_radius_px", 7))
    lw = float(cfg.viz.get("pred_point_linewidth", 1.5))
    show_probs = bool(cfg.viz.get("overlay_prob_on_test_images", True))
    digits = int(cfg.viz.get("prob_digits", 2))
    text_rgb = tuple(c / 255.0 for c in cfg.viz.get("prob_text_color", [255, 255, 0]))

    sx_raw, sy_raw = W_raw / float(rW), H_raw / float(rH)
    sx_gt, sy_gt = W_gt / float(rW), H_gt / float(rH)
    radius_raw = max(1, int(round(radius_base * (sx_raw + sy_raw) / 2.0)))
    radius_gt = max(1, int(round(radius_base * (sx_gt + sy_gt) / 2.0)))

    fig_w, fig_h = cfg.viz.get("collage_figsize", [12, 10])
    dpi = int(cfg.viz.get("collage_dpi", 140))
    title_fs = int(cfg.viz.get("collage_title_fontsize", 12))

    fig, axes = plt.subplots(2, 2, figsize=(fig_w, fig_h), dpi=dpi)
    for ax in axes.ravel(): ax.axis("off")

    _show_image(axes[0, 0], raw, title_input, title_fs)
    _show_image(axes[0, 1], gt_rgb, title_gt, title_fs)
    _show_image(axes[1, 0], raw, title_pred_on_input, title_fs)
    _draw_pred_points_on_axes(axes[1, 0], pred_on_raw, radius_raw, face_rgb, edge_rgb, lw, show_probs=show_probs, probs=pred_probs, digits=digits, text_rgb=text_rgb)
    _show_image(axes[1, 1], gt_rgb, title_pred_on_gt, title_fs)
    _draw_pred_points_on_axes(axes[1, 1], pred_on_gt, radius_gt, face_rgb, edge_rgb, lw, show_probs=show_probs, probs=pred_probs, digits=digits, text_rgb=text_rgb)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _save_slice_uncertainty_composite_matplotlib(
    cfg: Config,
    raw_path: str,
    mask_path: str,
    pred_centroids: List[Tuple[int, int]],
    gt_centroids: List[Tuple[int, int]],
    pred_probs: List[float],
    resized_hw: Tuple[int, int],
    pe_map01: Optional[np.ndarray],
    epi_map01: Optional[np.ndarray],
    unc_map01: Optional[np.ndarray],
    out_path: str,
):
    # Load raw and GT, compute overlays identical to 2x2 path
    raw = np.array(Image.open(raw_path).convert("RGB"))
    gt = np.array(Image.open(mask_path).convert("L"))
    gt_rgb = np.stack([gt, gt, gt], axis=-1)

    rH, rW = int(resized_hw[0]), int(resized_hw[1])
    H_raw, W_raw = raw.shape[0], raw.shape[1]
    H_gt, W_gt = gt_rgb.shape[0], gt_rgb.shape[1]

    pred_on_raw = _scale_points(pred_centroids, (rH, rW), (H_raw, W_raw))
    pred_on_gt = _scale_points(pred_centroids, (rH, rW), (H_gt, W_gt))

    rotate = bool(cfg.viz.get("rotate_cw_90", False))
    if rotate:
        raw = _rotate_if_needed(raw, True)
        gt_rgb = _rotate_if_needed(gt_rgb, True)
        pred_on_raw = _rotate_points_cw90(pred_on_raw, H_raw, W_raw)
        pred_on_gt = _rotate_points_cw90(pred_on_gt, H_gt, W_gt)
        H_raw, W_raw = raw.shape[0], raw.shape[1]
        H_gt, W_gt = gt_rgb.shape[0], gt_rgb.shape[1]
        # Rotate uncertainty maps to stay visually consistent
        if pe_map01 is not None:
            pe_map01 = _rotate_if_needed(pe_map01, True)
        if epi_map01 is not None:
            epi_map01 = _rotate_if_needed(epi_map01, True)
        if unc_map01 is not None:
            unc_map01 = _rotate_if_needed(unc_map01, True)

    overlay_mode = str(cfg.viz.get("overlay_orientation", "none")).lower().strip()
    if overlay_mode != "none":
        pred_on_raw = _apply_overlay_only_orientation(pred_on_raw, H_raw, W_raw, overlay_mode)
        pred_on_gt = _apply_overlay_only_orientation(pred_on_gt, H_gt, W_gt, overlay_mode)
    # Note: orientation is overlay-only; maps are not warped here by design.

    face_rgb = tuple(float(v) for v in cfg.viz.get("pred_point_color", [0, 1, 0]))
    edge_rgb = tuple(float(v) for v in cfg.viz.get("pred_point_edgecolor", [1, 1, 0]))
    radius_base = int(cfg.viz.get("pred_point_radius_px", 7))
    lw = float(cfg.viz.get("pred_point_linewidth", 1.5))
    show_probs = bool(cfg.viz.get("overlay_prob_on_test_images", True))
    digits = int(cfg.viz.get("prob_digits", 2))
    text_rgb = tuple(c / 255.0 for c in cfg.viz.get("prob_text_color", [255, 255, 0]))

    sx_raw, sy_raw = W_raw / float(rW), H_raw / float(rH)
    sx_gt, sy_gt = W_gt / float(rW), H_gt / float(rH)
    radius_raw = max(1, int(round(radius_base * (sx_raw + sy_raw) / 2.0)))
    radius_gt = max(1, int(round(radius_base * (sx_gt + sy_gt) / 2.0)))

    # Figure settings
    fig_w, fig_h = cfg.viz.get("uncertainty_composite_figsize", [12, 14])
    dpi = int(cfg.viz.get("uncertainty_composite_dpi", 140))
    title_fs = int(cfg.viz.get("uncertainty_title_fontsize", 12))
    cmap_name = str(cfg.uncertainty.get("save_maps", {}).get("colormap", "magma"))
    show_colorbar = bool(cfg.uncertainty.get("save_maps", {}).get("colorbar", True))

    fig, axes = plt.subplots(3, 2, figsize=(fig_w, fig_h), dpi=dpi)
    for ax in axes.ravel(): ax.axis("off")

    # Row 1: Raw | Pred on Input
    _show_image(axes[0, 0], raw, "Input", title_fs)
    _show_image(axes[0, 1], raw, "Prediction on Input", title_fs)
    _draw_pred_points_on_axes(axes[0, 1], pred_on_raw, radius_raw, face_rgb, edge_rgb, lw, show_probs=show_probs, probs=pred_probs, digits=digits, text_rgb=text_rgb)

    # Row 2: Pred on GT | Epistemic
    _show_image(axes[1, 0], gt_rgb, "Predicted Dots on GT", title_fs)
    _draw_pred_points_on_axes(axes[1, 0], pred_on_gt, radius_gt, face_rgb, edge_rgb, lw, show_probs=show_probs, probs=pred_probs, digits=digits, text_rgb=text_rgb)
    if epi_map01 is not None:
        _show_uncert_map(axes[1, 1], epi_map01, "Epistemic (MC Dropout MI)", title_fs, cmap_name, show_colorbar)
    else:
        _show_image(axes[1, 1], np.zeros((rH, rW), dtype=np.uint8), "Epistemic (disabled)", title_fs)

    # Row 3: PE | UNC
    if pe_map01 is not None:
        _show_uncert_map(axes[2, 0], pe_map01, "Predictive Entropy", title_fs, cmap_name, show_colorbar)
    else:
        _show_image(axes[2, 0], np.zeros((rH, rW), dtype=np.uint8), "Predictive Entropy (disabled)", title_fs)
    if unc_map01 is not None:
        _show_uncert_map(axes[2, 1], unc_map01, "Combined Uncertainty", title_fs, cmap_name, show_colorbar)
    else:
        _show_image(axes[2, 1], np.zeros((rH, rW), dtype=np.uint8), "Combined Uncertainty (disabled)", title_fs)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def evaluate_model_on_test(cfg: Config, hp: dict, model_path: str, patient_data: dict, test_ids: List[str], eval_tag: str, logger):
    _, val_tf = make_transforms(cfg, hp)

    test_items = []
    for pid in test_ids:
        test_items.extend(patient_data[pid])
    if len(test_items) == 0:
        raise RuntimeError("Test set is empty; cannot compute final metrics.")

    test_loader = DataLoader(Dataset(test_items, transform=val_tf), batch_size=1, shuffle=False, num_workers=int(cfg.train.get("num_workers", 0)), pin_memory=(cfg.device.type == "cuda"))

    model = create_model(cfg, hp)
    model.load_state_dict(torch.load(model_path, map_location=cfg.device))
    model.eval()

    amp_enabled = bool(cfg.train.get("use_amp", True)) and (cfg.device.type == "cuda")

    patient_predictions: Dict[str, List[dict]] = {}
    all_ed_mm: List[float] = []
    TP_total = FP_total = FN_total = 0

    base_tol = float(cfg.metrics["detection_tolerance_mm"])
    sweep_tols = [float(t) for t in cfg.metrics.get("tolerance_sweep_mm", [])]
    all_tols = list(sweep_tols)
    if base_tol not in all_tols:
        all_tols.insert(0, base_tol)

    patient_ed_map: Dict[str, List[float]] = {}
    per_patient_tol_counts: Dict[str, Dict[float, Dict[str, int]]] = {}
    per_slice_rows: List[dict] = []

    max_imgs_per_patient = cfg.viz.get("test_predictions_max_slices_per_patient", None)
    save_images = bool(cfg.viz.get("save_test_predictions", True))

    unc_cfg = cfg.uncertainty
    unc_enabled = bool(unc_cfg.get("enabled", True))
    mc_cfg = unc_cfg.get("mc_dropout", {})
    mc_enabled = bool(mc_cfg.get("enabled", False))
    mc_n = int(mc_cfg.get("num_samples", 15))
    mc_seed = mc_cfg.get("seed", None)
    epi_cfg = unc_cfg.get("epistemic", {})
    epi_enabled = bool(epi_cfg.get("enabled", True))
    epi_measure = str(epi_cfg.get("measure", "mi")).lower().strip()
    pe_enabled = bool(unc_cfg.get("predictive_entropy", {}).get("enabled", True))
    pool_cfg = unc_cfg.get("per_needle_pool", {})
    pool_w = int(pool_cfg.get("window_size", 11))
    pool_type = str(pool_cfg.get("pool_type", "mean")).lower().strip()
    pool_norm = str(pool_cfg.get("normalize", "minmax_slice")).lower().strip()
    comb_cfg = unc_cfg.get("combine", {})
    comb_enabled = bool(comb_cfg.get("enabled", True))
    w_pe = float(comb_cfg.get("weight_pe", 0.5))
    w_epi = float(comb_cfg.get("weight_epi", 0.5))
    norm_mode = str(comb_cfg.get("normalization", "minmax_slice")).lower().strip()
    save_maps_cfg = unc_cfg.get("save_maps", {})
    save_maps = bool(save_maps_cfg.get("enabled", True))
    save_pe_map = bool(save_maps_cfg.get("save_pe", True))
    save_epi_map = bool(save_maps_cfg.get("save_epi", True))
    save_comb_map = bool(save_maps_cfg.get("save_combined", True))
    map_fmt = str(save_maps_cfg.get("format", "png"))
    cmap_name = str(save_maps_cfg.get("colormap", "magma"))
    colorbar_on = bool(save_maps_cfg.get("colorbar", True))
    save_uncert_composite = bool(cfg.viz.get("save_uncertainty_composite", True))

    with torch.no_grad():
        for batch in test_loader:
            img = batch["image"].to(cfg.device, non_blocking=True)
            msk = batch["label"].to(cfg.device, non_blocking=True)

            pid = batch["patient_id"][0] if isinstance(batch["patient_id"], (list, tuple)) else batch["patient_id"]
            s_val = batch["slice_num"]
            s_val = s_val[0] if isinstance(s_val, (list, tuple)) else s_val
            slice_num = int(s_val.item()) if isinstance(s_val, torch.Tensor) else int(s_val)
            file_id, H_orig, W_orig = _get_img_info_from_batch(batch)

            with torch.amp.autocast("cuda", enabled=amp_enabled):
                logits = model(img)
                probs = torch.sigmoid(logits)
                thr = float(param(cfg, hp, "metrics.pred_threshold"))
                bin_pred = (probs > thr).float()

            metrics_img, matches, pred_centroids, gt_centroids = compute_euclidean_mm_and_matches(
                cfg, bin_pred[0], msk[0], file_id, (H_orig, W_orig), tuple(cfg.train["img_size"])
            )
            dists_mm = metrics_img["ED_mm_list"]
            n_pred = metrics_img["Needle_Count_Pred"]
            n_gt = metrics_img["Needle_Count_GT"]

            all_ed_mm.extend(dists_mm)
            TP_total += metrics_img["TP"]
            FP_total += metrics_img["FP"]
            FN_total += metrics_img["FN"]
            if dists_mm:
                patient_ed_map.setdefault(pid, []).extend(dists_mm)

            tol_stats_slice: Dict[str, Any] = {}
            for tol in all_tols:
                tp_t = sum(1 for d in dists_mm if d <= tol)
                fp_t = n_pred - tp_t
                fn_t = n_gt - tp_t
                precision_t = tp_t / (tp_t + fp_t) if (tp_t + fp_t) > 0 else 0.0
                recall_t = tp_t / (tp_t + fn_t) if (tp_t + fn_t) > 0 else 0.0
                f1_t = 2 * precision_t * recall_t / (precision_t + recall_t) if (precision_t + recall_t) > 0 else 0.0
                key_suffix = f"{tol:g}mm"
                tol_stats_slice.update(
                    {
                        f"TP_at_{key_suffix}": tp_t,
                        f"FP_at_{key_suffix}": fp_t,
                        f"FN_at_{key_suffix}": fn_t,
                        f"Unmatched_Pred_at_{key_suffix}": fp_t,
                        f"Unmatched_GT_at_{key_suffix}": fn_t,
                        f"Precision_at_{key_suffix}": precision_t,
                        f"Recall_at_{key_suffix}": recall_t,
                        f"F1_at_{key_suffix}": f1_t,
                    }
                )
            p_stats = per_patient_tol_counts.setdefault(pid, {})
            for tol in all_tols:
                agg_t = p_stats.setdefault(tol, {"TP": 0, "FP": 0, "FN": 0})
                agg_t["TP"] += tol_stats_slice[f"TP_at_{tol:g}mm"]
                agg_t["FP"] += tol_stats_slice[f"FP_at_{tol:g}mm"]
                agg_t["FN"] += tol_stats_slice[f"FN_at_{tol:g}mm"]

            per_slice_rows.append(
                {
                    "patient_id": str(pid),
                    "slice_num": slice_num,
                    "file_id": file_id,
                    "Needle_Count_Pred": n_pred,
                    "Needle_Count_GT": n_gt,
                    "Mean_Euclidean_Distance_mm": metrics_img["Mean_Euclidean_Distance_mm"],
                    "Max_Euclidean_Distance_mm": metrics_img["Max_Euclidean_Distance_mm"],
                    "Num_ED_pairs": len(dists_mm),
                    **tol_stats_slice,
                }
            )

            prob_map = probs[0, 0].detach().cpu().numpy()
            bin_np = bin_pred[0, 0].detach().cpu().numpy().astype(np.uint8)
            pred_probs = _extract_pred_probs(cfg, bin_np, prob_map, pred_centroids)

            pe_map = None; epi_map = None; unc_comb_map = None
            pred_pe_vals: List[float] = []; pred_epi_vals: List[float] = []; pred_unc_vals: List[float] = []

            if unc_enabled:
                prob_stack = None
                if mc_enabled and epi_enabled:
                    prob_stack = _mc_dropout_prob_stack(cfg=cfg, model=model, img_tensor=img, num_samples=mc_n, seed=mc_seed, amp_enabled=amp_enabled)

                if pe_enabled:
                    if prob_stack is not None:
                        mean_p = np.mean(prob_stack, axis=0)
                        pe_map = _entropy_from_prob_map(mean_p)
                    else:
                        pe_map = _entropy_from_prob_map(prob_map)

                if epi_enabled and prob_stack is not None:
                    if epi_measure == "variance":
                        epi_map = np.var(prob_stack, axis=0)
                    else:
                        mean_p = np.mean(prob_stack, axis=0)
                        H_mean = _entropy_from_prob_map(mean_p)
                        H_each = _entropy_from_prob_map(prob_stack)
                        epi_map = H_mean - np.mean(H_each, axis=0)
                elif epi_enabled and prob_stack is None:
                    epi_map = None

                if comb_enabled and (pe_map is not None or epi_map is not None):
                    def _norm_map_for_combine(arr: Optional[np.ndarray]) -> Optional[np.ndarray]:
                        if arr is None:
                            return None
                        if norm_mode == "minmax_slice":
                            return _minmax_norm_map(arr)
                        else:
                            return arr.astype(np.float32)
                    pe_n = _norm_map_for_combine(pe_map) if pe_map is not None else None
                    epi_n = _norm_map_for_combine(epi_map) if epi_map is not None else None
                    if pe_n is not None and epi_n is not None:
                        unc_comb_map = w_pe * pe_n + w_epi * epi_n
                    elif pe_n is not None:
                        unc_comb_map = pe_n * w_pe
                    elif epi_n is not None:
                        unc_comb_map = epi_n * w_epi

                if pred_centroids:
                    if pe_map is not None:
                        pred_pe_vals = _pool_windowed_values(pe_map, pred_centroids, pool_w, pool_type)
                    if epi_map is not None:
                        pred_epi_vals = _pool_windowed_values(epi_map, pred_centroids, pool_w, pool_type)

                    if pool_norm == "minmax_slice":
                        pe_vals_n = _minmax_norm_list(pred_pe_vals) if len(pred_pe_vals) > 0 else []
                        epi_vals_n = _minmax_norm_list(pred_epi_vals) if len(pred_epi_vals) > 0 else []
                    else:
                        pe_vals_n = [float(v) for v in pred_pe_vals]
                        epi_vals_n = [float(v) for v in pred_epi_vals]

                    pred_unc_vals = []
                    for i in range(len(pred_centroids)):
                        pe_i = pe_vals_n[i] if i < len(pe_vals_n) and pe_vals_n[i] == pe_vals_n[i] else 0.0
                        epi_i = epi_vals_n[i] if i < len(epi_vals_n) and epi_vals_n[i] == epi_vals_n[i] else 0.0
                        if comb_enabled:
                            pred_unc_vals.append(w_pe * pe_i + w_epi * epi_i)
                        else:
                            if pe_vals_n and not epi_vals_n:
                                pred_unc_vals.append(pe_i)
                            elif epi_vals_n and not pe_vals_n:
                                pred_unc_vals.append(epi_i)
                            else:
                                pred_unc_vals.append(0.0)

            img_path = batch["image_path"][0] if isinstance(batch["image_path"], (list, tuple)) else batch["image_path"]
            label_path = batch["label_path"][0] if isinstance(batch["label_path"], (list, tuple)) else batch["label_path"]
            out_dir = os.path.join(cfg.predictions_dir, eval_tag, str(pid))
            os.makedirs(out_dir, exist_ok=True)

            # Existing 2x2 composite (unchanged)
            if save_images:
                if (max_imgs_per_patient is None) or (len(patient_predictions.get(pid, [])) < int(max_imgs_per_patient)):
                    out_name = os.path.splitext(os.path.basename(img_path))[0] + ".png"
                    out_path = os.path.join(out_dir, out_name)
                    title_input = f"Input: {os.path.basename(img_path)}"
                    title_gt = f"Ground Truth: {os.path.basename(label_path)} | GT Needles: {len(gt_centroids)}"
                    title_pred_input = f"Prediction on Input | Pred Needles: {len(pred_centroids)}"
                    title_pred_on_gt = f"Predicted Dots on GT | Pred Needles: {len(pred_centroids)}"
                    _save_slice_collage_matplotlib(cfg, img_path, label_path, pred_centroids, gt_centroids, pred_probs, tuple(cfg.train["img_size"]), out_path, title_input, title_gt, title_pred_input, title_pred_on_gt)

            # Save color-mapped uncertainty maps
            if save_maps and unc_enabled:
                base_name = os.path.splitext(os.path.basename(img_path))[0]
                if pe_map is not None and save_pe_map:
                    pe_display = _minmax_norm_map(pe_map) if norm_mode == "minmax_slice" else pe_map.astype(np.float32)
                    _save_unc_map_colormap(os.path.join(out_dir, f"{base_name}_pe.{map_fmt}"), pe_display, cmap_name)
                if epi_map is not None and save_epi_map:
                    epi_display = _minmax_norm_map(epi_map) if norm_mode == "minmax_slice" else epi_map.astype(np.float32)
                    _save_unc_map_colormap(os.path.join(out_dir, f"{base_name}_epi.{map_fmt}"), epi_display, cmap_name)
                if unc_comb_map is not None and save_comb_map:
                    comb_display = _minmax_norm_map(unc_comb_map) if norm_mode == "minmax_slice" else unc_comb_map.astype(np.float32)
                    _save_unc_map_colormap(os.path.join(out_dir, f"{base_name}_unc.{map_fmt}"), comb_display, cmap_name)

                # Also save the requested 3x2 composite that includes uncertainty maps
                if save_uncert_composite:
                    # Normalize to [0,1] for display
                    pe_display = _minmax_norm_map(pe_map) if pe_map is not None else None
                    epi_display = _minmax_norm_map(epi_map) if epi_map is not None else None
                    comb_display = _minmax_norm_map(unc_comb_map) if unc_comb_map is not None else None
                    composite_path = os.path.join(out_dir, f"{base_name}_composite.{map_fmt}")
                    _save_slice_uncertainty_composite_matplotlib(
                        cfg,
                        img_path,
                        label_path,
                        pred_centroids,
                        gt_centroids,
                        pred_probs,
                        tuple(cfg.train["img_size"]),
                        pe_display,
                        epi_display,
                        comb_display,
                        composite_path,
                    )

            if pid not in patient_predictions:
                patient_predictions[pid] = []
            patient_predictions[pid].append(
                {
                    "slice_num": slice_num,
                    "pred_centroids": pred_centroids,
                    "pred_probs": pred_probs,
                    "pred_pe": pred_pe_vals,
                    "pred_epi": pred_epi_vals,
                    "pred_unc": pred_unc_vals,
                    "gt_centroids": gt_centroids,
                    "matches": matches,
                    "image_path": img_path,
                    "label_path": label_path,
                    "n_pred": n_pred,
                    "n_gt": n_gt,
                    "file_id": os.path.basename(img_path),
                }
            )

    for pid, slices in patient_predictions.items():
        slices_sorted = sorted(slices, key=lambda s: s["slice_num"], reverse=True)
        max_z = len(slices_sorted) - 1
        first_image_path = slices_sorted[0]["image_path"]
        H_orig, _W_orig = _get_original_hw_from_file(first_image_path)
        if H_orig:
            mm_per_pixel = cfg.metrics["pixel_to_mm"] * (H_orig / cfg.train["img_size"][0])
        else:
            mm_per_pixel = cfg.metrics["pixel_to_mm"]
        _ = save_needle_coordinates(cfg, str(pid), slices_sorted, max_z, eval_tag, mm_per_pixel)

    mean_ed = float(np.mean(all_ed_mm)) if all_ed_mm else float("nan")
    max_ed = float(np.max(all_ed_mm)) if all_ed_mm else float("nan")
    std_ed = float(np.std(all_ed_mm)) if all_ed_mm else float("nan")
    precision = TP_total / (TP_total + FP_total) if (TP_total + FP_total) > 0 else 0.0
    recall = TP_total / (TP_total + FN_total) if (TP_total + FN_total) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    final_csv = os.path.join(cfg.base_output_dir, f"{eval_tag}_final_metrics.csv")
    pd.DataFrame(
        [{"tag": eval_tag, "detection_tolerance_mm": base_tol, "mean_euclidean_mm": mean_ed, "max_euclidean_mm": max_ed, "std_euclidean_mm": std_ed, "precision": precision, "recall": recall, "f1": f1}]
    ).to_csv(final_csv, index=False)

    per_slice_csv = os.path.join(cfg.base_output_dir, f"{eval_tag}_per_slice_metrics.csv")
    pd.DataFrame(per_slice_rows).to_csv(per_slice_csv, index=False)

    patient_rows: List[dict] = []
    for pid, slices in patient_predictions.items():
        pid_str = str(pid)
        n_slices = len(slices)
        total_pred = sum(s["n_pred"] for s in slices)
        total_gt = sum(s["n_gt"] for s in slices)
        ed_list = patient_ed_map.get(pid, [])
        mean_ed_p = float(np.mean(ed_list)) if ed_list else float("nan")
        max_ed_p = float(np.max(ed_list)) if ed_list else float("nan")
        std_ed_p = float(np.std(ed_list)) if ed_list else float("nan")
        row = {"patient_id": pid_str, "n_slices": n_slices, "total_pred_needles": total_pred, "total_gt_needles": total_gt, "mean_ed_mm": mean_ed_p, "max_ed_mm": max_ed_p, "std_ed_mm": std_ed_p, "n_ed_pairs": len(ed_list)}
        for tol in all_tols:
            key_suffix = f"{tol:g}mm"
            stats_t = per_patient_tol_counts.get(pid, {}).get(tol, {"TP": 0, "FP": 0, "FN": 0})
            tp_t, fp_t, fn_t = stats_t["TP"], stats_t["FP"], stats_t["FN"]
            precision_t = tp_t / (tp_t + fp_t) if (tp_t + fp_t) > 0 else 0.0
            recall_t = tp_t / (tp_t + fn_t) if (tp_t + fn_t) > 0 else 0.0
            f1_t = 2 * precision_t * recall_t / (precision_t + recall_t) if (precision_t + recall_t) > 0 else 0.0
            row.update({f"TP_at_{key_suffix}": tp_t, f"FP_at_{key_suffix}": fp_t, f"FN_at_{key_suffix}": fn_t, f"Precision_at_{key_suffix}": precision_t, f"Recall_at_{key_suffix}": recall_t, f"F1_at_{key_suffix}": f1_t})
        patient_rows.append(row)

    per_patient_df = pd.DataFrame(patient_rows)
    per_patient_csv = os.path.join(cfg.base_output_dir, f"{eval_tag}_per_patient_metrics.csv")
    per_patient_df.to_csv(per_patient_csv, index=False)

    ci_conf = float(cfg.metrics.get("ci_confidence_level", 0.95))
    ci_rows = []
    ed_means_for_ci = [float(v) for v in per_patient_df["mean_ed_mm"].tolist() if isinstance(v, (float, int)) and math.isfinite(v)]
    mean_ed_over_pat, ci_low_ed, ci_high_ed = _mean_ci(ed_means_for_ci, ci_conf)
    ci_rows.append({"metric": "mean_ed_mm", "tol_mm": float("nan"), "mean_over_patients": mean_ed_over_pat, "ci_lower": ci_low_ed, "ci_upper": ci_high_ed, "n_patients": len(ed_means_for_ci), "confidence_level": ci_conf})

    for tol in all_tols:
        key_suffix = f"{tol:g}mm"
        for metric_name, col_pattern in [("Precision", "Precision_at_{}"), ("Recall", "Recall_at_{}"), ("F1", "F1_at_{}")]:
            col_name = col_pattern.format(key_suffix)
            if col_name not in per_patient_df.columns:
                continue
            vals = [float(v) for v in per_patient_df[col_name].tolist() if isinstance(v, (float, int)) and math.isfinite(v)]
            mean_m, ci_low_m, ci_high_m = _mean_ci(vals, ci_conf)
            ci_rows.append({"metric": metric_name, "tol_mm": tol, "mean_over_patients": mean_m, "ci_lower": ci_low_m, "ci_upper": ci_high_m, "n_patients": len(vals), "confidence_level": ci_conf})

    ci_df = pd.DataFrame(ci_rows)
    ci_csv = os.path.join(cfg.base_output_dir, f"{eval_tag}_per_patient_ci_summary.csv")
    ci_df.to_csv(ci_csv, index=False)

    logger = logging.getLogger("needle_train")
    logger.info(f"[{eval_tag}] Test: Mean ED(mm): {mean_ed:.3f} | P: {precision:.3f} | R: {recall:.3f} | F1: {f1:.3f}", extra={"milestone": True})

    return {"mean_ed_mm": mean_ed, "precision": precision, "recall": recall, "f1": f1, "csv": final_csv, "per_slice_csv": per_slice_csv, "per_patient_csv": per_patient_csv, "per_patient_ci_csv": ci_csv}


# ----
# Main
# ----

def main():
    parser = argparse.ArgumentParser(description="Needle Segmentation Training (bayes_opt via Optuna)")
    parser.add_argument("--config", type=str, required=True, help="Path to JSON config file")
    args = parser.parse_args()

    cfg_json = load_config_json(args.config)
    cfg = Config(cfg_json)

    logger = setup_logger(cfg)
    install_exception_hook(logger)

    logger.info(f"Using device: {cfg.device}", extra={"milestone": True})
    if cfg.description:
        logger.info(f"Description: {cfg.description}")

    patient_data = get_patient_data_dicts(cfg, logger)
    if not patient_data:
        raise ValueError("No valid data found")
    all_patient_ids = list(patient_data.keys())

    strat_labels = compute_patient_strat_labels(patient_data) if cfg.cv.get("stratify_by_patient", True) else {pid: 1 for pid in all_patient_ids}
    strat_all = [strat_labels[pid] for pid in all_patient_ids]
    train_val_ids, test_ids = train_test_split(all_patient_ids, test_size=cfg.cv["test_size"], random_state=cfg.cv["random_state"], stratify=strat_all)
    logger.info(f"Train+Val patients: {len(train_val_ids)} | Test patients: {len(test_ids)}")

    if cfg.search.get("enabled", True):
        train_ids = train_val_ids
        strategy = str(cfg.search.get("strategy", "bayes_opt")).lower().strip()
        if strategy not in ("bayes_opt", "bayes"):
            raise ValueError("Only 'bayes_opt' strategy is supported. Please set search.strategy to 'bayes_opt' in config.")
        best_trial, _ = bayes_opt_search(cfg, logger, patient_data, train_ids, strat_labels)

        best_hp = best_trial["params"] if best_trial else {}
        best_hp = sanitize_hyperparams(cfg, best_hp, logger)

        y_train = np.array([strat_labels[pid] for pid in train_ids])
        tr_ids, va_ids = train_test_split(train_ids, test_size=0.15, random_state=cfg.cv["random_state"], stratify=y_train)
        train_loader, val_loader = build_loaders_from_ids(cfg, best_hp, patient_data, tr_ids, va_ids)

        final_dir = os.path.join(cfg.search_dir, f"final_best_model_bayes_opt")
        os.makedirs(final_dir, exist_ok=True)
        logger.info("[Final] Training best model on a single train/val split (Fold 0) before testing.", extra={"milestone": True})
        best_metric, best_epoch, final_model_path, epoch_csv, best_val_metrics = train_one_fold(cfg, best_hp, train_loader, val_loader, fold_idx=0, fold_dir=final_dir, logger=logger, epoch_csv_outdir=os.path.join(cfg.epoch_csv_dir, f"final_bayes_opt"))
        save_json(os.path.join(final_dir, "final_best_model_metrics.json"), {"best_epoch": best_epoch, "best_metric_mean_ed_mm": best_metric, "best_val_metrics": best_val_metrics, "model_path": final_model_path})
        _ = evaluate_model_on_test(cfg, best_hp, final_model_path, patient_data, test_ids, eval_tag=f"bayes_opt_best", logger=logger)
        logger.info("Bayes_opt workflow complete.", extra={"milestone": True})
        return


if __name__ == "__main__":

    main()
