# Tracking comparison

Both runs use identical detector outputs and source timestamps.

| Metric | Legacy | Motion |
|---|---:|---:|
| Frames with selected armor pose | 1609 | 2280 |
| Selected-pose coverage | 51.08% | 72.38% |
| EKF measurement updates | 1595 | 2263 |
| Updates / selected poses | 99.13% | 99.25% |
| EKF rejection events | 17 | 17 |
| Filter segment resets | 5 | 217 |
| Explicit visual-track switches | 0 | 216 |
| Median reprojection error (px) | 0.3137 | 0.3476 |

Same cached post-deduplication detector inputs, no yaw or permanent physical-identity ground truth. Camera yaw and angle steps include actual rotation. Switches split EKF segments; a lower reset count is not an identity-accuracy metric.
