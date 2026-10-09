# Source image timestamp. Pose and angles use header.frame_id:
# camera_optical_frame (camera mode) or odom (world mode).
std_msgs/Header header
# A geometrically valid detection/PnP pose exists in this image.
bool detection_valid
# The current measurement passed EKF gating, or initialized the filter.
bool measurement_used
bool filter_initialized
# The yaw filter started a new segment, including same-track reinitialization.
bool target_reset
uint32 track_id
# Temporary image-motion identity; 0 when absent or in legacy association mode.
uint32 visual_track_id
# A different visual track was selected and the yaw filter was reset.
bool target_switched
# Neural class ID, not a physical plate identity. -1 when missing.
int32 class_id
float64 detection_confidence
# Radians. Camera mode: outward normal heading atan2(n_x, -n_z).
# Odom mode: world yaw atan2(n_y, n_x), positive toward world +Y.
# observed_yaw is NaN if no valid detection; filtered values are NaN before init.
float64 observed_yaw
float64 filtered_yaw
float64 yaw_rate
float64 yaw_variance
float64 reprojection_error
# Armor frame: X outward normal, Y left, Z up; pose is in header.frame_id.
# Translation is in meters. Ignore pose when detection_valid is false.
geometry_msgs/Pose pose
# Image pixel corners: top-left, bottom-left, bottom-right, top-right.
geometry_msgs/Point32[4] corners
