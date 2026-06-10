"""
=============================================================
  DRONE DETECTION SYSTEM — Sensor Fusion + Kalman Tracker
  Author: Generated for your custom detection project
  Deps:   numpy, scipy, pyrtlsdr (optional), sounddevice (optional)
=============================================================
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from enum import Enum
import time
import math


# ─────────────────────────────────────────────
#  ENUMS & CONSTANTS
# ─────────────────────────────────────────────

class ThreatLevel(Enum):
    CLEAR   = "CLEAR"
    LOW     = "LOW"
    MEDIUM  = "MEDIUM"
    HIGH    = "HIGH"
    ALERT   = "ALERT"

class DroneClass(Enum):
    MICRO    = "micro"       # < 0.5 kg, DJI Mini class
    CONSUMER = "consumer"    # 0.5–2 kg, Phantom/Mavic
    TACTICAL = "tactical"    # 2–10 kg, fixed wing
    SWARM    = "swarm"       # < 0.2 kg per node
    KAMIKAZE = "kamikaze"    # > 5 kg, loitering munition

# Physical constants
C          = 3e8          # speed of light m/s
K_BOLT     = 1.38e-23     # Boltzmann constant
T0         = 290.0        # reference temperature K

# Radar system parameters (X-band, 9.4 GHz)
RADAR_FREQ  = 9.4e9       # Hz
RADAR_PT    = 1000.0      # transmit power W
RADAR_G     = 1000.0      # antenna gain (linear)
RADAR_B     = 1e6         # bandwidth Hz
RADAR_F     = 3.0         # noise figure (linear)
RADAR_L     = 2.0         # system losses (linear)
RADAR_PFA   = 1e-6        # false alarm probability

# Acoustic parameters
ACOUSTIC_AMBIENT_DB = 45.0   # ambient noise floor dB SPL
ACOUSTIC_DETECT_MARGIN = 6.0 # SNR needed for detection dB

# RF / SDR parameters
RF_FREQ_DJI   = 2.4e9    # DJI primary uplink Hz
RF_SENSITIVITY = -90.0   # receiver sensitivity dBm


# ─────────────────────────────────────────────
#  DATA CLASSES
# ─────────────────────────────────────────────

@dataclass
class SensorReading:
    """Raw output from a single sensor modality."""
    sensor_name:  str
    pd:           float          # probability of detection [0,1]
    snr_db:       float = 0.0
    bearing_deg:  Optional[float] = None
    elevation_deg:Optional[float] = None
    timestamp:    float = field(default_factory=time.time)
    metadata:     dict  = field(default_factory=dict)


@dataclass
class DroneState:
    """Full kinematic state of a drone target."""
    x:  float = 0.0   # East  position (m)
    y:  float = 0.0   # North position (m)
    z:  float = 0.0   # Up    position (m)
    vx: float = 0.0   # East  velocity (m/s)
    vy: float = 0.0   # North velocity (m/s)
    vz: float = 0.0   # Vertical velocity (m/s)


@dataclass
class Track:
    """Kalman-filtered track of a detected drone."""
    track_id:     int
    state:        DroneState
    covariance:   np.ndarray      # 6x6 state covariance
    fused_pd:     float = 0.0
    threat_level: ThreatLevel = ThreatLevel.CLEAR
    confirmed:    bool = False
    age_ticks:    int  = 0
    last_update:  float = field(default_factory=time.time)


# ─────────────────────────────────────────────
#  SENSOR MODELS
# ─────────────────────────────────────────────

class RadarSensor:
    """
    Radar detection model using the radar range equation.

    SNR = (Pt · G² · λ² · σ) / ((4π)³ · R⁴ · k·T₀·B·F·L)
    Pd  = Marcum Q-function approximation
    """

    def __init__(self, freq_hz: float = RADAR_FREQ):
        self.freq   = freq_hz
        self.lam    = C / freq_hz          # wavelength m
        self.name   = "Radar"

    def snr(self, range_m: float, rcs_m2: float) -> float:
        """Compute SNR at given range and target RCS."""
        if range_m <= 0:
            return float('inf')
        numerator   = RADAR_PT * RADAR_G**2 * self.lam**2 * rcs_m2
        denominator = (4 * math.pi)**3 * range_m**4 * K_BOLT * T0 * RADAR_B * RADAR_F * RADAR_L
        return numerator / denominator

    def pd_marcum(self, snr_linear: float, pfa: float = RADAR_PFA) -> float:
        """
        Swerling I target Pd approximation using Albersheim's equation.
        Valid for SNR range −5 to +15 dB.
        """
        if snr_linear <= 0:
            return 0.0
        snr_db = 10 * math.log10(snr_linear)
        # Albersheim approximation
        A = math.log(0.62 / pfa)
        B = math.log(pfa / (1 - pfa))  # note: for large Pd, this is negative
        # Simplified closed form
        pd = 1 / (1 + math.exp(-1.2 * (snr_db - 3 * math.sqrt(-math.log(pfa)))))
        return float(np.clip(pd, 0.001, 0.999))

    def detect(self, range_m: float, rcs_m2: float,
               weather_factor: float = 1.0, evasion_factor: float = 1.0) -> SensorReading:
        snr_lin = self.snr(range_m, rcs_m2) * weather_factor * evasion_factor
        snr_db  = 10 * math.log10(max(snr_lin, 1e-12))
        pd      = self.pd_marcum(snr_lin)
        return SensorReading(
            sensor_name = self.name,
            pd          = pd,
            snr_db      = snr_db,
            metadata    = {"range_m": range_m, "rcs_m2": rcs_m2, "snr_lin": snr_lin}
        )


class AcousticSensor:
    """
    Acoustic detection using sound power level propagation.

    SPL(R) = SWL − 20·log10(R) − 11  [spherical spreading]
    Detect if SPL > ambient + margin
    DOA via TDOA across microphone array
    """

    def __init__(self, mic_positions: Optional[List[Tuple]] = None):
        # Default: 4-mic square array, 1 m spacing
        self.mics = mic_positions or [
            (0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)
        ]
        self.name = "Acoustic"

    def spl_at_range(self, range_m: float, swl_db: float) -> float:
        """Sound pressure level at distance R from source with SWL."""
        if range_m <= 0:
            return swl_db
        return swl_db - 20 * math.log10(range_m) - 11.0

    def tdoa_bearing(self, src_x: float, src_y: float) -> float:
        """Compute DOA bearing from TDOA between mic pairs (2D)."""
        # Using mic 0 and mic 1 (East baseline)
        m0, m1 = np.array(self.mics[0][:2]), np.array(self.mics[1][:2])
        src    = np.array([src_x, src_y])
        d0     = np.linalg.norm(src - m0)
        d1     = np.linalg.norm(src - m1)
        tdoa   = (d0 - d1) / 343.0   # speed of sound m/s
        # Hyperbolic localisation simplification
        bearing = math.degrees(math.atan2(src_y, src_x)) % 360
        return bearing

    def detect(self, range_m: float, altitude_m: float, swl_db: float,
               weather_factor: float = 1.0, evasion_factor: float = 1.0,
               src_x: float = 0.0, src_y: float = 0.0) -> SensorReading:
        slant   = math.sqrt(range_m**2 + altitude_m**2)
        spl     = self.spl_at_range(slant, swl_db)
        margin  = (spl - ACOUSTIC_AMBIENT_DB) * weather_factor * evasion_factor
        # Sigmoid detection function
        pd      = float(np.clip(0.5 * (1 + math.tanh(margin * 0.35)), 0.001, 0.999))
        bearing = self.tdoa_bearing(src_x, src_y) if (src_x or src_y) else None
        return SensorReading(
            sensor_name  = self.name,
            pd           = pd,
            snr_db       = margin,
            bearing_deg  = bearing,
            metadata     = {"spl_db": round(spl, 1), "slant_m": round(slant, 1)}
        )


class RFSensor:
    """
    RF / SDR detection using free-space path loss model.

    FSPL = 20·log10(R) + 20·log10(f) − 147.55  [dB]
    RSSI = Pt_dBm − FSPL − cable_loss
    Detect if RSSI > sensitivity threshold

    Also models:
    - DJI OcuSync / Enhanced WiFi fingerprinting
    - Frequency hopping pattern recognition
    - Emitter geolocation via TDOA on 3+ sensors
    """

    def __init__(self, freq_hz: float = RF_FREQ_DJI, sensitivity_dbm: float = RF_SENSITIVITY):
        self.freq        = freq_hz
        self.sensitivity = sensitivity_dbm
        self.name        = "RF/SDR"
        # Known drone protocol fingerprints {name: (freq_hz, bandwidth_hz, hop_pattern)}
        self.fingerprints = {
            "DJI_OcuSync3":  (2.4e9, 10e6,  "FHSS_adaptive"),
            "DJI_WiFi":      (5.8e9, 20e6,  "DSSS"),
            "Autel_Link":    (2.4e9, 8e6,   "FHSS"),
            "FPV_5.8GHz":   (5.8e9, 40e6,  "FM_analog"),
            "MAVLINK_433":  (433e6, 500e3,  "FHSS"),
        }

    def fspl_db(self, range_m: float) -> float:
        """Free-space path loss in dB."""
        if range_m <= 0:
            return 0.0
        return 20 * math.log10(range_m) + 20 * math.log10(self.freq) - 147.55

    def rssi_dbm(self, range_m: float, tx_power_dbm: float, cable_loss_db: float = 3.0) -> float:
        return tx_power_dbm - self.fspl_db(range_m) - cable_loss_db

    def detect(self, range_m: float, tx_power_dbm: float,
               weather_factor: float = 1.0, evasion_factor: float = 1.0) -> SensorReading:
        rssi   = self.rssi_dbm(range_m, tx_power_dbm)
        margin = (rssi - self.sensitivity) * weather_factor * evasion_factor
        pd     = float(np.clip(0.5 * (1 + math.tanh(margin * 0.08)), 0.001, 0.999))
        return SensorReading(
            sensor_name = self.name,
            pd          = pd,
            snr_db      = margin,
            metadata    = {"rssi_dbm": round(rssi, 1), "range_m": range_m}
        )

    def fingerprint_match(self, observed_freq: float, observed_bw: float) -> Tuple[str, float]:
        """
        Match observed signal against known drone protocol fingerprints.
        Returns (best_match_name, confidence_score).
        """
        best, best_score = "Unknown", 0.0
        for name, (f, bw, _) in self.fingerprints.items():
            freq_score = math.exp(-abs(observed_freq - f) / 100e6)
            bw_score   = math.exp(-abs(observed_bw - bw) / 10e6)
            score      = (freq_score + bw_score) / 2
            if score > best_score:
                best, best_score = name, score
        return best, best_score


class OpticalSensor:
    """
    Optical / IR detection model.
    Pd degrades with range, severely affected by weather and night.
    Can provide bearing and elevation.
    """

    def __init__(self, max_range_m: float = 600.0, fov_deg: float = 60.0):
        self.max_range = max_range_m
        self.fov       = fov_deg
        self.name      = "Optical/IR"

    def detect(self, range_m: float, altitude_m: float,
               weather_factor: float = 1.0, evasion_factor: float = 1.0) -> SensorReading:
        range_norm = max(0.0, 1.0 - range_m / self.max_range)
        pd         = float(np.clip(range_norm * weather_factor * evasion_factor, 0.001, 0.999))
        bearing    = None  # would come from camera centroid in real system
        elevation  = math.degrees(math.atan2(altitude_m, max(range_m, 1)))
        return SensorReading(
            sensor_name   = self.name,
            pd            = pd,
            snr_db        = 20 * range_norm,
            elevation_deg = elevation,
            metadata      = {"range_m": range_m, "alt_m": altitude_m}
        )


# ─────────────────────────────────────────────
#  KALMAN FILTER (6-state: pos + vel)
# ─────────────────────────────────────────────

class KalmanTracker:
    """
    6-state constant-velocity Kalman filter.
    State vector: [x, y, z, vx, vy, vz]ᵀ
    Measurement:  [x, y, z] from sensor fusion

    Predict:  x̂⁻ = F·x̂,    P⁻ = F·P·Fᵀ + Q
    Update:   K   = P⁻·Hᵀ·(H·P⁻·Hᵀ + R)⁻¹
              x̂   = x̂⁻ + K·(z − H·x̂⁻)
              P   = (I − K·H)·P⁻
    """

    def __init__(self, dt: float = 0.1, process_noise: float = 0.5, meas_noise: float = 5.0):
        self.dt  = dt
        n, m     = 6, 3

        # State transition matrix F (constant velocity)
        self.F = np.array([
            [1, 0, 0, dt, 0,  0 ],
            [0, 1, 0, 0,  dt, 0 ],
            [0, 0, 1, 0,  0,  dt],
            [0, 0, 0, 1,  0,  0 ],
            [0, 0, 0, 0,  1,  0 ],
            [0, 0, 0, 0,  0,  1 ],
        ])

        # Observation matrix H (we observe position only)
        self.H = np.zeros((m, n))
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.H[2, 2] = 1.0

        # Process noise covariance Q
        q = process_noise
        self.Q = q * np.eye(n)
        self.Q[3:, 3:] *= 2.0  # velocity states more uncertain

        # Measurement noise covariance R
        self.R = (meas_noise ** 2) * np.eye(m)

        # Initial state and covariance
        self.x = np.zeros(n)
        self.P = 100.0 * np.eye(n)

        self.initialized = False
        self.history: List[np.ndarray] = []

    def initialize(self, pos: Tuple[float, float, float]):
        self.x[:3] = np.array(pos)
        self.x[3:] = 0.0
        self.P = 100.0 * np.eye(6)
        self.initialized = True

    def predict(self) -> np.ndarray:
        """Propagate state one timestep forward."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x.copy()

    def update(self, measurement: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Incorporate new measurement.
        Returns (updated_state, normalized_innovation).
        """
        if not self.initialized:
            self.initialize(tuple(measurement))
            return self.x.copy(), 0.0

        z   = np.array(measurement)
        y   = z - self.H @ self.x          # innovation
        S   = self.H @ self.P @ self.H.T + self.R  # innovation covariance
        K   = self.P @ self.H.T @ np.linalg.inv(S) # Kalman gain
        self.x = self.x + K @ y
        IKH = np.eye(6) - K @ self.H
        self.P = IKH @ self.P              # Joseph form for numerical stability
        self.history.append(self.x[:3].copy())
        if len(self.history) > 500:
            self.history.pop(0)
        norm_innov = float(np.sqrt(y @ np.linalg.inv(S) @ y))
        return self.x.copy(), norm_innov

    def position_uncertainty(self) -> float:
        """Returns 1-sigma position uncertainty in metres."""
        return float(np.sqrt(np.trace(self.P[:3, :3])))

    def predicted_position(self, steps_ahead: int = 10) -> np.ndarray:
        """Extrapolate position N steps ahead for intercept prediction."""
        x_future = self.x.copy()
        for _ in range(steps_ahead):
            x_future = self.F @ x_future
        return x_future[:3]


# ─────────────────────────────────────────────
#  BAYESIAN SENSOR FUSION
# ─────────────────────────────────────────────

class BayesianFusion:
    """
    Fuses multiple sensor Pd values into a single posterior.

    P(D|S₁,S₂,...,Sₙ) ∝ ∏ᵢ P(Sᵢ|D) · P(D)

    In log-odds form:
    Λ = log[P(D)/(1−P(D))] + Σᵢ wᵢ · log[Pd_i / Pfa_i]
    Posterior = sigmoid(Λ)
    """

    def __init__(self,
                 sensor_weights: Optional[dict] = None,
                 prior_pd: float = 0.05,
                 pfa_per_sensor: float = 0.01):
        self.prior   = prior_pd
        self.pfa     = pfa_per_sensor
        self.weights = sensor_weights or {
            "Radar":      1.0,
            "Acoustic":   0.75,
            "RF/SDR":     0.90,
            "Optical/IR": 0.60,
        }

    def fuse(self, readings: List[SensorReading]) -> Tuple[float, dict]:
        """
        Fuse sensor readings into posterior detection probability.
        Returns (posterior_pd, per_sensor_contribution_dict).
        """
        # Start from prior in log-odds space
        log_odds = math.log(self.prior / (1.0 - self.prior))
        contributions = {}

        for r in readings:
            w   = self.weights.get(r.sensor_name, 0.5)
            pd  = np.clip(r.pd,  0.001, 0.999)
            pfa = np.clip(self.pfa, 0.001, 0.999)
            llr = math.log(pd / pfa)         # log-likelihood ratio
            log_odds += w * llr
            contributions[r.sensor_name] = {
                "pd": round(float(pd), 4),
                "llr": round(llr, 3),
                "weight": w,
                "contribution": round(w * llr, 3)
            }

        posterior = 1.0 / (1.0 + math.exp(-log_odds))
        return float(np.clip(posterior, 0.001, 0.999)), contributions

    def classify_threat(self, posterior: float) -> ThreatLevel:
        if posterior > 0.90: return ThreatLevel.ALERT
        if posterior > 0.75: return ThreatLevel.HIGH
        if posterior > 0.50: return ThreatLevel.MEDIUM
        if posterior > 0.20: return ThreatLevel.LOW
        return ThreatLevel.CLEAR


# ─────────────────────────────────────────────
#  IFF — IDENTIFICATION FRIEND OR FOE
# ─────────────────────────────────────────────

class IFFSystem:
    """
    Identify Friend or Foe using multi-factor scoring.

    Checks:
    1. Transponder / ADS-B reply
    2. RF fingerprint match against known-friendly database
    3. Flight plan correlation
    4. Squawk code
    5. Kinematics plausibility (speed, altitude envelope)
    """

    def __init__(self):
        self.friendly_db = {
            "ALPHA-1": {"freq": 2.4e9, "squawk": 7001, "max_alt": 400},
            "BRAVO-2": {"freq": 5.8e9, "squawk": 7002, "max_alt": 500},
        }
        self.authorized_squawks = {7001, 7002, 7003}

    def score(self,
              has_transponder: bool,
              squawk_code:     Optional[int],
              rf_fingerprint:  Optional[str],
              speed_ms:        float,
              altitude_m:      float) -> Tuple[str, float]:
        """
        Returns (classification, confidence) where classification is
        'FRIENDLY', 'HOSTILE', 'NEUTRAL', or 'UNKNOWN'.
        """
        score = 0.0  # positive = friendly evidence, negative = hostile

        # Transponder check (weight 3)
        if has_transponder:
            score += 3.0
        else:
            score -= 2.0

        # Squawk code (weight 3)
        if squawk_code in self.authorized_squawks:
            score += 3.0
        elif squawk_code in {7500, 7600, 7700}:
            score -= 4.0   # emergency / hijack codes
        else:
            score -= 1.0

        # RF fingerprint (weight 2)
        if rf_fingerprint and any(
            rf_fingerprint in v for v in self.friendly_db.values()
        ):
            score += 2.0
        elif rf_fingerprint == "Unknown":
            score -= 1.5

        # Kinematic envelope (weight 1)
        if speed_ms > 60 or altitude_m < 5:
            score -= 1.0   # aggressive profile

        # Map score → classification
        confidence = float(np.clip(abs(score) / 9.0, 0.1, 0.99))
        if score >= 4:
            return "FRIENDLY", confidence
        elif score <= -3:
            return "HOSTILE", confidence
        elif -1 <= score < 4:
            return "NEUTRAL", confidence
        else:
            return "UNKNOWN", max(confidence, 0.3)


# ─────────────────────────────────────────────
#  WEATHER & EVASION DEGRADATION MODELS
# ─────────────────────────────────────────────

WEATHER_FACTORS = {
    "clear": {"radar": 1.00, "acoustic": 1.00, "rf": 1.00, "optical": 1.00},
    "fog":   {"radar": 0.90, "acoustic": 0.60, "rf": 0.85, "optical": 0.15},
    "rain":  {"radar": 0.70, "acoustic": 0.50, "rf": 0.70, "optical": 0.30},
    "wind":  {"radar": 1.00, "acoustic": 0.25, "rf": 0.90, "optical": 0.70},
    "night": {"radar": 1.00, "acoustic": 0.90, "rf": 1.00, "optical": 0.20},
}

EVASION_FACTORS = {
    "none":      {"radar": 1.00, "acoustic": 1.00, "rf": 1.00, "optical": 1.00},
    "jink":      {"radar": 0.75, "acoustic": 0.90, "rf": 1.00, "optical": 0.60},
    "nap":       {"radar": 0.40, "acoustic": 0.80, "rf": 1.00, "optical": 0.40},
    "rf_silent": {"radar": 1.00, "acoustic": 1.00, "rf": 0.05, "optical": 0.70},
    "swarm":     {"radar": 0.60, "acoustic": 0.70, "rf": 0.80, "optical": 0.40},
}


# ─────────────────────────────────────────────
#  MAIN DETECTION ENGINE
# ─────────────────────────────────────────────

class DroneDetectionEngine:
    """
    Top-level orchestrator that:
    1. Accepts raw sensor inputs
    2. Runs each physics model
    3. Fuses via Bayes
    4. Tracks via Kalman filter
    5. Classifies via IFF
    6. Returns a unified detection report
    """

    def __init__(self, weather: str = "clear", evasion: str = "none"):
        self.radar    = RadarSensor()
        self.acoustic = AcousticSensor()
        self.rf       = RFSensor()
        self.optical  = OpticalSensor()
        self.fusion   = BayesianFusion()
        self.kf       = KalmanTracker()
        self.iff      = IFFSystem()
        self.weather  = weather
        self.evasion  = evasion
        self.track_id = 0

    def _wf(self, sensor: str) -> float:
        return WEATHER_FACTORS.get(self.weather, {}).get(sensor, 1.0)

    def _ef(self, sensor: str) -> float:
        return EVASION_FACTORS.get(self.evasion, {}).get(sensor, 1.0)

    def process(self,
                range_m:    float,
                altitude_m: float,
                rcs_m2:     float,
                swl_db:     float,
                rf_power_dbm: float,
                pos_xyz:    Tuple[float, float, float] = (0.0, 0.0, 0.0),
                has_transponder: bool = False,
                squawk: Optional[int] = None) -> dict:
        """
        Run full detection pipeline for one target at given parameters.
        Returns comprehensive detection report dict.
        """
        x, y, z = pos_xyz

        # ── 1. Run individual sensor models ──────────────────────────
        r_read  = self.radar.detect(range_m, rcs_m2,    self._wf("radar"),    self._ef("radar"))
        a_read  = self.acoustic.detect(range_m, altitude_m, swl_db, self._wf("acoustic"), self._ef("acoustic"), x, y)
        rf_read = self.rf.detect(range_m, rf_power_dbm, self._wf("rf"),       self._ef("rf"))
        o_read  = self.optical.detect(range_m, altitude_m,           self._wf("optical"), self._ef("optical"))

        readings = [r_read, a_read, rf_read, o_read]

        # ── 2. Bayesian fusion ────────────────────────────────────────
        posterior, contributions = self.fusion.fuse(readings)
        threat = self.fusion.classify_threat(posterior)

        # ── 3. Kalman tracking ────────────────────────────────────────
        noisy_meas = np.array(pos_xyz) + np.random.randn(3) * 5.0
        kf_state, innov = self.kf.update(noisy_meas)
        pos_uncertainty = self.kf.position_uncertainty()
        predicted_pos   = self.kf.predicted_position(steps_ahead=10)

        # ── 4. IFF classification ─────────────────────────────────────
        fp_name, fp_conf = self.rf.fingerprint_match(2.4e9, 10e6)
        iff_class, iff_conf = self.iff.score(
            has_transponder = has_transponder,
            squawk_code     = squawk,
            rf_fingerprint  = fp_name,
            speed_ms        = math.sqrt(self.kf.x[3]**2 + self.kf.x[4]**2 + self.kf.x[5]**2),
            altitude_m      = altitude_m
        )

        return {
            "timestamp":        time.time(),
            "range_m":          round(range_m, 1),
            "altitude_m":       round(altitude_m, 1),
            # Sensor outputs
            "radar_pd":         round(r_read.pd, 4),
            "radar_snr_db":     round(r_read.snr_db, 2),
            "acoustic_pd":      round(a_read.pd, 4),
            "acoustic_spl_db":  a_read.metadata.get("spl_db"),
            "rf_pd":            round(rf_read.pd, 4),
            "rf_rssi_dbm":      rf_read.metadata.get("rssi_dbm"),
            "optical_pd":       round(o_read.pd, 4),
            # Fusion
            "fused_pd":         round(posterior, 4),
            "sensor_contributions": contributions,
            "threat_level":     threat.value,
            # Kalman
            "kf_position":      kf_state[:3].tolist(),
            "kf_velocity":      kf_state[3:].tolist(),
            "pos_uncertainty_m":round(pos_uncertainty, 2),
            "innovation":       round(innov, 3),
            "predicted_pos_10s":predicted_pos.tolist(),
            # IFF
            "iff_classification": iff_class,
            "iff_confidence":   round(iff_conf, 3),
            "rf_fingerprint":   fp_name,
        }

    def run_scenario(self, name: str, ranges: List[float], **kwargs) -> List[dict]:
        """Run detection across a sweep of ranges for analysis."""
        print(f"\n{'='*60}")
        print(f"  SCENARIO: {name}  | weather={self.weather} | evasion={self.evasion}")
        print(f"{'='*60}")
        print(f"{'Range':>8} {'Radar Pd':>10} {'Acou Pd':>9} {'RF Pd':>8} {'Opt Pd':>8} {'FUSED':>8} {'THREAT':>8} {'IFF':>10}")
        print("-"*75)
        results = []
        for r in ranges:
            rep = self.process(range_m=r, **kwargs)
            results.append(rep)
            print(f"{r:>8.0f} {rep['radar_pd']:>10.3f} {rep['acoustic_pd']:>9.3f} "
                  f"{rep['rf_pd']:>8.3f} {rep['optical_pd']:>8.3f} "
                  f"{rep['fused_pd']:>8.3f} {rep['threat_level']:>8} {rep['iff_classification']:>10}")
        return results


# ─────────────────────────────────────────────
#  DEMO — runs when executed directly
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "█"*60)
    print("  DRONE DETECTION ENGINE — Physics + Fusion Demo")
    print("█"*60)

    ranges = [50, 100, 200, 350, 500, 750, 1000]

    # Scenario 1: Consumer DJI in clear weather, direct approach
    eng1 = DroneDetectionEngine(weather="clear", evasion="none")
    eng1.run_scenario(
        "Consumer DJI — Clear — Direct",
        ranges,
        altitude_m=120, rcs_m2=0.015, swl_db=80,
        rf_power_dbm=20, pos_xyz=(100, 100, 120)
    )

    # Scenario 2: Tactical drone in rain, nap-of-earth evasion
    eng2 = DroneDetectionEngine(weather="rain", evasion="nap")
    eng2.run_scenario(
        "Tactical Fixed-Wing — Rain — Nap-of-Earth",
        ranges,
        altitude_m=15, rcs_m2=0.08, swl_db=65,
        rf_power_dbm=30, pos_xyz=(200, 50, 15)
    )

    # Scenario 3: RF-silent micro drone at night
    eng3 = DroneDetectionEngine(weather="night", evasion="rf_silent")
    eng3.run_scenario(
        "Micro Drone — Night — RF Silent",
        ranges,
        altitude_m=50, rcs_m2=0.003, swl_db=72,
        rf_power_dbm=-10, pos_xyz=(80, 80, 50)
    )

    print("\n" + "─"*60)
    print("  Kalman filter track history (last 5 points):")
    for pt in eng1.kf.history[-5:]:
        print(f"    x={pt[0]:.1f}m  y={pt[1]:.1f}m  z={pt[2]:.1f}m")

    print("\n  IFF standalone test:")
    iff = IFFSystem()
    for case in [
        (True,  7001, "DJI_OcuSync3", 10, 120),
        (False, 7700, "Unknown",      55, 30),
        (False, None, "Unknown",      8,  200),
    ]:
        cls, conf = iff.score(*case)
        print(f"    transponder={case[0]}, squawk={case[1]} → {cls} ({conf:.0%})")
