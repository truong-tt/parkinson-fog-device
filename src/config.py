"""Config: env var > YAML > in-code default."""

import os
import yaml
import json

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(SRC_DIR, '..'))

DATA_DIR = os.environ.get('HOPEGAIT_DATA_DIR', os.path.join(BASE_DIR, 'data'))
RAW_DATA_DIR = os.environ.get('HOPEGAIT_RAW_DATA_DIR', os.path.join(DATA_DIR, 'raw'))
PROCESSED_DATA_DIR = os.environ.get('HOPEGAIT_PROCESSED_DATA_DIR', os.path.join(DATA_DIR, 'processed'))
MODELS_DIR = os.environ.get('HOPEGAIT_MODELS_DIR', os.path.join(BASE_DIR, 'models'))

DEFAULT_CONFIG_PATH = os.path.join(BASE_DIR, 'config', 'training_config.yaml')
CONFIG_PATH = os.environ.get('HOPEGAIT_CONFIG_PATH', DEFAULT_CONFIG_PATH)

if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, 'r') as f:
        _cfg = yaml.safe_load(f) or {}
else:
    _cfg = {}


def get_config(env_name, yaml_key, default, parser=None):
    """Resolve a setting with precedence env var > YAML > default.

    Args:
        env_name: Environment variable name (highest precedence).
        yaml_key: Key in the YAML config file.
        default: Fallback value.
        parser: Optional callable to coerce the env-var string.

    Returns:
        The resolved configuration value.
    """
    env_val = os.environ.get(env_name)
    if env_val is not None:
        if parser:
            try:
                return parser(env_val)
            except Exception:
                pass
        return env_val
    return _cfg.get(yaml_key, default)


# --- Signal / feature pipeline ---
# Defaults match the upstream stanfordnmbl/imu-fog-detection dataset:
#   FREQ_SAMPLED = 128 Hz, FREQ_DESIRED = 64 Hz, WINDOW_DUR = 2 s
# After resampling to FREQ_DESIRED, a 2-second window is 128 samples.
RAW_SAMPLING_RATE = get_config('HOPEGAIT_RAW_SAMPLING_RATE', 'raw_sampling_rate', 128.0, parser=float)
SAMPLING_RATE = get_config('HOPEGAIT_SAMPLING_RATE', 'sampling_rate', 64.0, parser=float)
WINDOW_SIZES = get_config('HOPEGAIT_WINDOW_SIZES', 'window_sizes', [128], parser=json.loads)
WINDOW_OVERLAP = get_config('HOPEGAIT_WINDOW_OVERLAP', 'window_overlap', 0.5, parser=float)
# Rotate each recording so its mean gravity aligns to a canonical axis, removing
# per-recording mounting tilt that otherwise leaks through the gravity channels
# and breaks LOSO folds for inconsistently-mounted subjects (e.g. subject 6).
CANON_ORIENT = get_config('HOPEGAIT_CANON_ORIENT', 'canonicalize_orientation', True,
                          parser=lambda v: str(v).lower() in ('1', 'true', 'yes'))

# --- Model architecture ---
# 4-block TCN with dilations (1, 2, 4, 8) gives a receptive field of
#   rf = 1 + 2 * (k-1) * sum(dilations) = 1 + 4 * 15 = 61 samples
# At 64 Hz that's ~0.95 s — covers nearly half a 2 s window of past context.
NUM_INPUTS = get_config('HOPEGAIT_NUM_INPUTS', 'num_inputs', 9, parser=int)
NUM_CLASSES = get_config('HOPEGAIT_NUM_CLASSES', 'num_classes', 2, parser=int)
NUM_CHANNELS = get_config('HOPEGAIT_NUM_CHANNELS', 'num_channels', [32, 64, 96, 128], parser=json.loads)
KERNEL_SIZE = get_config('HOPEGAIT_KERNEL_SIZE', 'kernel_size', 3, parser=int)
DROPOUT = get_config('HOPEGAIT_DROPOUT', 'dropout', 0.3, parser=float)

# --- Training ---
SEED = get_config('HOPEGAIT_SEED', 'seed', 42, parser=int)
BATCH_SIZE = get_config('HOPEGAIT_BATCH_SIZE', 'batch_size', 64, parser=int)
EPOCHS = get_config('HOPEGAIT_EPOCHS', 'epochs', 50, parser=int)
LEARNING_RATE = get_config('HOPEGAIT_LEARNING_RATE', 'learning_rate', 1e-3, parser=float)
WEIGHT_DECAY = get_config('HOPEGAIT_WEIGHT_DECAY', 'weight_decay', 1e-4, parser=float)
CLASS_WEIGHTS = get_config('HOPEGAIT_CLASS_WEIGHTS', 'class_weights', [0.2, 0.8], parser=json.loads)
FOCAL_GAMMA = get_config('HOPEGAIT_FOCAL_GAMMA', 'focal_gamma', 2.0, parser=float)
EARLY_STOP_PATIENCE = get_config('HOPEGAIT_EARLY_STOP_PATIENCE', 'early_stop_patience', 10, parser=int)
LR_PATIENCE = get_config('HOPEGAIT_LR_PATIENCE', 'lr_patience', 5, parser=int)
NUM_WORKERS = get_config('HOPEGAIT_NUM_WORKERS', 'num_workers', 0, parser=int)

# --- Regularization / dense head ---
DROP_PATH = get_config('HOPEGAIT_DROP_PATH', 'drop_path', 0.1, parser=float)
USE_SE = get_config('HOPEGAIT_USE_SE', 'use_se', True, parser=lambda v: str(v).lower() in ('1', 'true', 'yes'))
DENSE_LOSS_WEIGHT = get_config('HOPEGAIT_DENSE_LOSS_WEIGHT', 'dense_loss_weight', 0.5, parser=float)
EMA_DECAY = get_config('HOPEGAIT_EMA_DECAY', 'ema_decay', 0.99, parser=float)  # 0.999 too slow for short runs -> chance-level EMA shadow (decision_log 2026-06-17)

# --- Augmentation ---
ROTATION_MAX_DEG = get_config('HOPEGAIT_ROTATION_MAX_DEG', 'rotation_max_deg', 15.0, parser=float)
ROTATION_PROB = get_config('HOPEGAIT_ROTATION_PROB', 'rotation_prob', 0.5, parser=float)

# --- Inference post-processing ---
# smooth_window is in predictions; at 1 Hz any averaging erases the ~4 s median
# FoG episode. 1 = pass-through (no smoothing) — won the 64 Hz sweep at band 0.05
# (per-subject MCC +0.197 vs +0.189 raw); larger values only cost detection.
SMOOTH_WINDOW = get_config('HOPEGAIT_SMOOTH_WINDOW', 'smooth_window', 1, parser=int)
# Band = HIGH - LOW feeds the asymmetric exit (exit at threshold - band). 0.20 was
# tuned for the old 100 Hz model; 0.05 won the 64 Hz sweep — confirm via sweep_postproc.
HYSTERESIS_LOW = get_config('HOPEGAIT_HYSTERESIS_LOW', 'hysteresis_low', 0.55, parser=float)
HYSTERESIS_HIGH = get_config('HOPEGAIT_HYSTERESIS_HIGH', 'hysteresis_high', 0.6, parser=float)

# --- Cloud / runtime ---
USE_AMP = get_config('HOPEGAIT_USE_AMP', 'use_amp', True, parser=lambda v: str(v).lower() in ('1', 'true', 'yes'))
DEVICE = get_config('HOPEGAIT_DEVICE', 'device', 'auto')