#pragma once
#include <Eigen/Dense>
#include <opencv2/core.hpp>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace armor_ekf {
struct ImageObservation {
  cv::Point2f center;
  cv::Size2f size;
  int class_id;
  double confidence;
};
struct ImageTrack {
  uint32_t id;
  Eigen::Vector4d state; // center x/y and velocity x/y, in pixels and pixels/s.
  Eigen::Matrix4d covariance;
  cv::Size2f size;
  int class_id, hits = 1, detection_index = -1;
  double confidence, last_seen;
  cv::Point2f center() const { return {float(state[0]), float(state[1])}; }
};
struct ImageAssociation {
  int selected = -1;
  uint32_t target_id = 0;
  bool target_changed = false;
  bool target_switched = false;
  std::vector<uint32_t> detection_ids;
  cv::Point2f predicted_center{};
  double gate = 0;
};

// Image-space identities are supported only while spatial motion is observable.
// Model classes are a weak matching cue: several physical plates share a class.
class MotionTracker {
public:
  MotionTracker(double timeout = 0.3, double gate_scale = 0.8, double switch_timeout = 0.08)
  : timeout_(timeout), gate_scale_(gate_scale), switch_timeout_(switch_timeout) {
    if (!(timeout > 0 && gate_scale > 0 && switch_timeout > 0 && switch_timeout <= timeout))
      throw std::invalid_argument("Invalid tracker parameters");
  }
  void clear() { tracks_.clear(); target_id_ = 0; last_stamp_ = std::numeric_limits<double>::quiet_NaN(); }
  const std::vector<ImageTrack> & tracks() const { return tracks_; }
  ImageAssociation update(const std::vector<ImageObservation> & observations, double stamp) {
    if (!std::isfinite(stamp)) throw std::invalid_argument("Invalid tracker timestamp");
    if (std::isfinite(last_stamp_) && (stamp <= last_stamp_ || stamp-last_stamp_ > timeout_)) clear();
    const double dt = std::isfinite(last_stamp_) ? stamp-last_stamp_ : 0;
    last_stamp_ = stamp;
    tracks_.erase(std::remove_if(tracks_.begin(), tracks_.end(), [&](const auto & t) {
      return stamp-t.last_seen > timeout_;
    }), tracks_.end());
    for (auto & t : tracks_) {
      Eigen::Matrix4d F = Eigen::Matrix4d::Identity(); F(0,2) = dt; F(1,3) = dt;
      Eigen::Matrix<double,4,2> G = Eigen::Matrix<double,4,2>::Zero();
      G(0,0) = G(1,1) = dt*dt/2; G(2,0) = G(3,1) = dt;
      t.state = F*t.state;
      t.covariance = F*t.covariance*F.transpose() + 4000000.0*G*G.transpose();
      t.detection_index = -1;
    }
    ImageAssociation result;
    result.detection_ids.assign(observations.size(), 0);
    // Rectangular minimum-cost assignment, with one unmatched dummy per track.
    std::vector<std::vector<double>> costs(tracks_.size(), std::vector<double>(observations.size()+tracks_.size(),1.0));
    for (size_t i = 0; i < tracks_.size(); ++i) {
      const auto & t = tracks_[i];
      if (t.id == target_id_) { result.predicted_center = t.center(); result.gate = gate(t,stamp); }
      for (size_t j = 0; j < observations.size(); ++j) {
        const auto & o = observations[j];
        const double distance = cv::norm(o.center-t.center());
        const double wr = o.size.width/std::max(1.0f,t.size.width);
        const double hr = o.size.height/std::max(1.0f,t.size.height);
        if (!(wr > 0 && hr > 0) || distance > gate(t,stamp) || hr < 0.5 || hr > 2.0) {
          costs[i][j] = 1e6; continue;
        }
        costs[i][j] = std::pow(distance/gate(t,stamp),2) +
          0.08*(std::pow(std::log(wr),2)+std::pow(std::log(hr),2)) + (o.class_id==t.class_id ? 0 : 0.04);
      }
    }
    const auto assignment = assign(costs);
    for (size_t i = 0; i < tracks_.size(); ++i) {
      const int j = assignment[i];
      if (j < 0 || j >= int(observations.size()) || costs[i][j] >= 1.0) continue;
      auto & t = tracks_[i]; const auto & o = observations[j];
      Eigen::Matrix<double,2,4> H = Eigen::Matrix<double,2,4>::Zero(); H(0,0) = H(1,1) = 1;
      const double sigma = std::max(2.0,double(o.size.height)*0.06);
      const Eigen::Matrix2d R = Eigen::Matrix2d::Identity()*sigma*sigma;
      const Eigen::Matrix2d S = H*t.covariance*H.transpose()+R;
      const Eigen::Matrix<double,4,2> K = t.covariance*H.transpose()*S.ldlt().solve(Eigen::Matrix2d::Identity());
      t.state += K*(Eigen::Vector2d(o.center.x,o.center.y)-H*t.state);
      const Eigen::Matrix4d A = Eigen::Matrix4d::Identity()-K*H;
      t.covariance = A*t.covariance*A.transpose()+K*R*K.transpose();
      t.size = o.size; t.class_id = o.class_id; t.confidence = o.confidence;
      t.last_seen = stamp; t.detection_index = j; ++t.hits;
      result.detection_ids[j] = t.id;
    }
    for (size_t j = 0; j < observations.size(); ++j) {
      if (result.detection_ids[j]) continue;
      const auto & o = observations[j];
      ImageTrack t;
      t.id = ++next_id_; t.state << o.center.x,o.center.y,0,0;
      t.covariance = Eigen::Matrix4d::Identity();
      t.covariance(0,0) = t.covariance(1,1) = 4;
      t.covariance(2,2) = t.covariance(3,3) = 1000000;
      t.size = o.size; t.class_id = o.class_id; t.confidence = o.confidence;
      t.last_seen = stamp; t.detection_index = int(j);
      tracks_.push_back(t); result.detection_ids[j] = t.id;
    }
    const auto primary = std::find_if(tracks_.begin(),tracks_.end(),[&](const auto & t){return t.id==target_id_;});
    const bool primary_hold = primary != tracks_.end() &&
      (primary->detection_index >= 0 || stamp-primary->last_seen <= switch_timeout_);
    if (primary_hold) {
      result.selected = primary->detection_index;
    } else {
      const ImageTrack * best = nullptr;
      for (const auto & t : tracks_) {
        if (t.detection_index < 0) continue;
        if (target_id_ && t.hits < 2) continue;
        if (!best || t.confidence > best->confidence) best = &t;
      }
      if (best) {
        result.target_changed = best->id != target_id_;
        result.target_switched = target_id_ != 0 && result.target_changed;
        target_id_ = best->id; result.selected = best->detection_index;
        result.predicted_center = best->center(); result.gate = gate(*best,stamp);
      } else if (primary == tracks_.end()) target_id_ = 0;
    }
    result.target_id = target_id_;
    return result;
  }
private:
  double gate(const ImageTrack & t, double stamp) const {
    const double base = std::max(40.0, t.size.width*gate_scale_);
    return base + std::min(base*0.5,std::max(0.0,stamp-t.last_seen-0.025)*100.0);
  }
  static std::vector<int> assign(const std::vector<std::vector<double>> & costs) {
    const int n = int(costs.size()); if (!n) return {};
    const int m = int(costs.front().size());
    std::vector<double> u(n+1),v(m+1);
    std::vector<int> p(m+1),way(m+1);
    for (int i = 1; i <= n; ++i) {
      p[0] = i; int j0 = 0;
      std::vector<double> minv(m+1,std::numeric_limits<double>::infinity());
      std::vector<bool> used(m+1,false);
      do {
        used[j0] = true; int i0 = p[j0],j1 = 0;
        double delta = std::numeric_limits<double>::infinity();
        for (int j = 1; j <= m; ++j) if (!used[j]) {
          const double cur = costs[i0-1][j-1]-u[i0]-v[j];
          if (cur < minv[j]) { minv[j] = cur; way[j] = j0; }
          if (minv[j] < delta) { delta = minv[j]; j1 = j; }
        }
        for (int j = 0; j <= m; ++j) {
          if (used[j]) { u[p[j]] += delta; v[j] -= delta; } else minv[j] -= delta;
        }
        j0 = j1;
      } while (p[j0] != 0);
      do { int j1 = way[j0]; p[j0] = p[j1]; j0 = j1; } while (j0);
    }
    std::vector<int> result(n,-1);
    for (int j = 1; j <= m; ++j) if (p[j]) result[p[j]-1] = j-1;
    return result;
  }
  std::vector<ImageTrack> tracks_;
  uint32_t next_id_ = 0, target_id_ = 0;
  double timeout_,gate_scale_,switch_timeout_,last_stamp_ = std::numeric_limits<double>::quiet_NaN();
};
} // namespace armor_ekf
