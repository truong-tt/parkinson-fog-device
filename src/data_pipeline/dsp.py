"""DSP utilities: causal filters and per-window frequency features."""

import numpy as np
from scipy.signal import butter, lfilter, lfilter_zi, welch, stft
from scipy.interpolate import interp1d


def _warm_lfilter(b, a, data):
    # Warm-start initial conditions with data[0] so the filter behaves as if
    # it had been running forever — kills the ~20-sample startup transient.
    if data.ndim == 1:
        zi = lfilter_zi(b, a) * data[0]
        y, _ = lfilter(b, a, data, zi=zi)
        return y
    out = np.empty_like(data)
    zi_unit = lfilter_zi(b, a)
    for c in range(data.shape[1]):
        col = data[:, c]
        y, _ = lfilter(b, a, col, zi=zi_unit * col[0])
        out[:, c] = y
    return out


def gravity_align_matrix(mean_gravity, target=(-1.0, 0.0, 0.0)):
    """Rotation mapping the mean-gravity direction onto ``target``.

    Canonicalizes sensor mounting: applied (as ``v @ R.T``) to a recording's
    acc/gravity/gyro it removes the absolute tilt, so recordings and subjects
    mounted at different orientations share one frame. Without it the gravity
    channels leak per-recording orientation that does not generalize across a
    LOSO split (e.g. a subject remounted up to ~180 deg apart between visits).

    Args:
        mean_gravity: Mean gravity vector ``(3,)`` of the recording.
        target: Canonical gravity direction (default ``-x``, matching the
            consistently-mounted subjects).

    Returns:
        ``(3, 3)`` float32 rotation; identity if gravity is degenerate.
    """
    g = np.asarray(mean_gravity, dtype=np.float64)
    n = np.linalg.norm(g)
    if n < 1e-8:
        return np.eye(3, dtype=np.float32)
    a = g / n
    b = np.asarray(target, dtype=np.float64)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if c > 1.0 - 1e-8:                      # already aligned
        return np.eye(3, dtype=np.float32)
    if c < -1.0 + 1e-8:                     # antiparallel: 180 deg about any perp axis
        perp = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        axis = np.cross(a, perp)
        axis = axis / np.linalg.norm(axis)
        K = np.array([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]])
        return (np.eye(3) + 2.0 * (K @ K)).astype(np.float32)
    vx = np.array([[0, -v[2], v[1]],
                   [v[2], 0, -v[0]],
                   [-v[1], v[0], 0]])
    R = np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + c))
    return R.astype(np.float32)


class IMUFilter:
    # Default fs matches the upstream FREQ_DESIRED (post-resample). At 64 Hz
    # nyquist is 32 Hz, so the 15 Hz lowpass and 0.3 Hz gravity cutoff are
    # comfortably below it.
    def __init__(self, fs=64.0, lowpass_hz=15.0, gravity_cutoff_hz=0.3, order=4,
                 raw_fs=None):
        self.fs = float(fs)
        self.raw_fs = float(raw_fs) if raw_fs else None
        self.nyq = 0.5 * self.fs
        self.b_lp, self.a_lp = butter(order, lowpass_hz / self.nyq, btype='low')
        self.b_grav, self.a_grav = butter(order, gravity_cutoff_hz / self.nyq, btype='low')
        # Anti-alias lowpass built at the RAW rate, applied BEFORE downsampling.
        # 15 Hz is well under the 64 Hz target Nyquist (32 Hz), so one filter
        # both band-limits and stops content >32 Hz folding back during resample.
        if self.raw_fs and self.raw_fs > self.fs:
            self.b_aa, self.a_aa = butter(order, lowpass_hz / (0.5 * self.raw_fs), btype='low')
        else:
            self.b_aa = self.a_aa = None

    def resample_data(self, timestamps, data):
        # Uniform-grid resample so training and MCU see identical fs regardless
        # of dropped packets or jittery sensor timestamps.
        dt = 1.0 / self.fs
        target = np.arange(timestamps[0], timestamps[-1], dt)
        f = interp1d(timestamps, data, axis=0, kind='linear', fill_value='extrapolate')
        return f(target), target

    def apply_lowpass(self, data):
        return _warm_lfilter(self.b_lp, self.a_lp, data)

    def apply_gravity_filter(self, data):
        return _warm_lfilter(self.b_grav, self.a_grav, data)

    def process_signal(self, acc, gyro, timestamps=None):
        """Resample (if timestamps given), lowpass, and split gravity from motion.

        Returns:
            ``(linear_acc, gravity, gyro_lp, new_timestamps)``; ``new_timestamps``
            is ``None`` when no input timestamps were supplied.
        """
        new_ts = None
        pre_filtered = False
        if timestamps is not None:
            # Anti-alias at the raw rate FIRST, then resample, so the 128->64
            # decimation can't alias HF content into the freeze band.
            if self.b_aa is not None:
                acc = _warm_lfilter(self.b_aa, self.a_aa, acc)
                gyro = _warm_lfilter(self.b_aa, self.a_aa, gyro)
                pre_filtered = True
            acc, new_ts = self.resample_data(timestamps, acc)
            gyro, _ = self.resample_data(timestamps, gyro)
        # Skip the redundant second lowpass when already band-limited pre-resample;
        # otherwise this is the only lowpass (no-timestamp / no-raw_fs path).
        if pre_filtered:
            acc_lp, gyro_lp = acc, gyro
        else:
            acc_lp = self.apply_lowpass(acc)
            gyro_lp = self.apply_lowpass(gyro)
        gravity = self.apply_gravity_filter(acc_lp)
        linear_acc = acc_lp - gravity
        return linear_acc, gravity, gyro_lp, new_ts


def freeze_index_window(linear_acc_window, fs=64.0, nperseg=None,
                        loco_band=(0.5, 3.0), freeze_band=(3.0, 8.0)):
    """Freeze Index: freeze-band power over locomotor-band power (Bächlin 2010).

    PSD is computed per-axis then summed. Do NOT take ``||acc||`` first — the
    magnitude rectifies sinusoids and doubles their apparent frequency,
    inverting the index.

    Args:
        linear_acc_window: ``(T, k)`` gravity-removed acceleration window.
        fs: Sampling rate in Hz.
        nperseg: Welch segment length; defaults to ``min(128, T)``.
        loco_band: Locomotor band ``(lo, hi)`` in Hz.
        freeze_band: Freeze band ``(lo, hi)`` in Hz.

    Returns:
        Scalar freeze index; ``0.0`` if the window is too short.
    """
    data = np.asarray(linear_acc_window)
    n = nperseg if nperseg is not None else min(128, len(data))
    if n < 8:
        return 0.0
    f, Pxx = welch(data, fs=fs, nperseg=n, axis=0)
    loco = float(np.sum(Pxx[(f >= loco_band[0]) & (f <= loco_band[1])]))
    freeze = float(np.sum(Pxx[(f >= freeze_band[0]) & (f <= freeze_band[1])]))
    return freeze / (loco + 1e-6)


def stft_band_power_window(linear_acc_window, fs=64.0, nperseg=64, band=(3.0, 8.0)):
    """Mean STFT magnitude within ``band`` for a window.

    Args:
        linear_acc_window: ``(T, k)`` gravity-removed acceleration window.
        fs: Sampling rate in Hz.
        nperseg: STFT segment length (clamped to the window length).
        band: Frequency band ``(lo, hi)`` in Hz.

    Returns:
        Scalar mean band magnitude; ``0.0`` if the window is too short.
    """
    data = np.asarray(linear_acc_window)
    n = min(nperseg, len(data))
    if n < 8:
        return 0.0
    f, _, Zxx = stft(data, fs=fs, nperseg=n, noverlap=n // 2, axis=0)
    mask = (f >= band[0]) & (f <= band[1])
    if not mask.any():
        return 0.0
    return float(np.mean(np.abs(Zxx[mask])))


class RobustScaler:
    """Median/IQR scaler, fit on training fold only and persisted with the model."""

    def __init__(self, median=None, iqr=None):
        self.median = None if median is None else np.asarray(median, dtype=np.float32)
        self.iqr = None if iqr is None else np.asarray(iqr, dtype=np.float32)

    def fit(self, data):
        if data.ndim == 3:
            flat = data.reshape(-1, data.shape[-1])
        elif data.ndim == 2:
            flat = data
        else:
            raise ValueError(f"Expected 2D or 3D data, got shape {data.shape}")
        self.median = np.median(flat, axis=0).astype(np.float32)
        q75, q25 = np.percentile(flat, [75, 25], axis=0)
        iqr = q75 - q25
        self.iqr = np.where(iqr < 1e-6, 1.0, iqr).astype(np.float32)
        return self

    def transform(self, data):
        if self.median is None or self.iqr is None:
            raise RuntimeError("RobustScaler must be fit (or loaded) before transform.")
        return ((data - self.median) / self.iqr).astype(np.float32)

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)

    def save(self, path):
        np.savez(path, median=self.median, iqr=self.iqr)

    @classmethod
    def load(cls, path):
        arr = np.load(path)
        return cls(median=arr['median'], iqr=arr['iqr'])
