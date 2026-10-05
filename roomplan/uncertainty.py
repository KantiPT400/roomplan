"""Error model. All intervals are reported as 95% (+-1.96 sigma).

Every measurement gets sigma^2 = random^2 + systematic^2 + drift^2:
  random     : standard error of the fitted plane(s), from the points themselves
  systematic : sensor scale/bias. Per-tier constants, see TIER_SYSTEMATIC. These are priors from the
               literature/device class, NOT calibrated against ground truth in this repo (none was
               available for the sample data). docs/report.md says so explicitly.
  drift      : pose drift accumulated between the frames that observed the two ends of a measurement.
               Estimated per capture from loop-closure residuals (see drift.py).
"""
from __future__ import annotations
import numpy as np

Z95 = 1.96

# (absolute m, relative fraction) per tier
TIER_SYSTEMATIC = {
    "lidar": (0.005, 0.004),
    "video": (0.010, 0.020),   # metric scale comes from LiDAR-free priors, see video tier
    "photo": (0.020, 0.040),
}
UNSUPPORTED_EDGE_SIGMA = 0.10   # edge not backed by a measured wall plane (bounded by furniture / unseen)


def length_sigma(L, se_a, se_b, tier="lidar", drift=0.0, supported=True):
    a, r = TIER_SYSTEMATIC[tier]
    rnd2 = se_a ** 2 + se_b ** 2
    if not supported:
        rnd2 += UNSUPPORTED_EDGE_SIGMA ** 2
    return float(np.sqrt(rnd2 + a ** 2 + (r * L) ** 2 + drift ** 2))


def ci(value, sigma, nd=3):
    return {"value": round(float(value), nd), "ci95": [round(float(value - Z95 * sigma), nd),
                                                        round(float(value + Z95 * sigma), nd)],
            "sigma": round(float(sigma), 4)}
