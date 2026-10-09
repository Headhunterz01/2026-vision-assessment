#pragma once
#include "armor_ekf/pipeline.hpp"
#include <ostream>
#include <iomanip>
namespace armor_ekf {
inline void writeResultHeader(std::ostream & csv) {
  csv << "frame,time,detection_valid,measurement_used,filter_initialized,target_reset,track_id,class_id,confidence,observed_yaw,filtered_yaw,yaw_rate,yaw_variance,reprojection_error,x,y,z,model_rows,confidence_class_pass,valid_boxes,in_bounds_corners,valid_corner_sets,deduplicated_detections,pnp_attempts,pnp_solutions,physical_pose_rejects,reprojection_rejects,valid_poses,association_rejected,association_min_distance_px,association_gate_px,tracked_class_id,nearest_candidate_class_id,nearest_candidate_confidence,nearest_candidate_matches_track,ekf_rejected,visual_track_id,target_switched,predicted_center_x,predicted_center_y,detection_track_ids,selected_detection_index,frame_id,camera_yaw,camera_x,camera_y,camera_z,qx,qy,qz,qw,camera_qx,camera_qy,camera_qz,camera_qw,world_yaw_rejects\n";
  csv << std::setprecision(12);
}
inline void writeResult(std::ostream & csv, int index, double stamp, const Result & r) {
  csv << index << ',' << stamp << ',' << r.detection_valid << ',' << r.measurement_used << ','
      << r.filter_initialized << ',' << r.target_reset << ',' << r.track_id << ',' << r.class_id << ','
      << r.confidence << ',' << r.observed_yaw << ',' << r.filtered_yaw << ',' << r.yaw_rate << ','
      << r.yaw_variance << ',' << r.reprojection_error << ',' << r.output_tvec[0] << ',' << r.output_tvec[1] << ',' << r.output_tvec[2] << ','
      << r.model_rows << ',' << r.confidence_class_pass << ',' << r.valid_boxes << ','
      << r.in_bounds_corners << ',' << r.valid_corner_sets << ',' << r.deduplicated_detections << ','
      << r.pnp_attempts << ',' << r.pnp_solutions << ',' << r.physical_pose_rejects << ','
      << r.reprojection_rejects << ',' << r.valid_poses << ',' << r.association_rejected << ','
      << r.association_min_distance_px << ',' << r.association_gate_px << ','
      << r.tracked_class_id << ',' << r.nearest_candidate_class_id << ','
      << r.nearest_candidate_confidence << ',' << r.nearest_candidate_matches_track << ','
      << r.ekf_rejected << ',' << r.visual_track_id << ',' << r.target_switched << ','
      << r.predicted_center.x << ',' << r.predicted_center.y << ',';
  for (size_t i = 0; i < r.detection_track_ids.size(); ++i) {
    if (i) csv << ';';
    csv << r.detection_track_ids[i];
  }
  const Eigen::Quaterniond q(rotationFromRvec(r.output_rvec)), camera_q(rotationFromRvec(r.rvec));
  csv << ',' << r.selected_detection_index << ',' << r.frame_id << ',' << r.camera_yaw
      << ',' << r.tvec[0] << ',' << r.tvec[1] << ',' << r.tvec[2]
      << ',' << q.x() << ',' << q.y() << ',' << q.z() << ',' << q.w()
      << ',' << camera_q.x() << ',' << camera_q.y() << ',' << camera_q.z() << ',' << camera_q.w()
      << ',' << r.world_yaw_rejects << '\n';
}
}
