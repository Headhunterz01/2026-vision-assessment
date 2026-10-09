#pragma once
#include "armor_ekf/reprojector.hpp"
#include <ostream>
#include <iomanip>

namespace armor_ekf {
inline void writeProjectionHeader(std::ostream & out) {
  out << "frame,time,valid,plate,depth_valid,in_image,front_facing,center_x,center_y,center_z,pdf_yaw";
  for (const auto * frame : {"world", "camera"})
    for (const auto * field : {"x","y","z","qx","qy","qz","qw"}) out << ',' << frame << '_' << field;
  for (int k=0;k<4;++k) out << ",k" << k << "x,k" << k << 'y';
  out << '\n' << std::setprecision(17);
}
inline void writeProjection(std::ostream & out, int frame, double stamp, const RobotProjection & p) {
  if (!p.valid) {
    out << frame << ',' << stamp << ",0,-1,0,0,0";
    for (int k=0;k<26;++k) out << ",nan";
    out << '\n'; return;
  }
  for (int i=0;i<4;++i) {
    const auto & a=p.plates[i];
    out << frame << ',' << stamp << ",1," << i << ',' << a.depth_valid << ',' << a.in_image << ',' << a.front_facing;
    for (int k=0;k<3;++k) out << ',' << p.center[k];
    out << ',' << a.pdf_yaw;
    for (const auto * pose : {&a.world_pose,&a.camera_pose}) {
      for (int k=0;k<3;++k) out << ',' << pose->translation()[k];
      const Eigen::Quaterniond q(pose->linear());
      out << ',' << q.x() << ',' << q.y() << ',' << q.z() << ',' << q.w();
    }
    for (const auto & pixel : a.pixels) out << ',' << pixel.x << ',' << pixel.y;
    out << '\n';
  }
}
} // namespace armor_ekf
