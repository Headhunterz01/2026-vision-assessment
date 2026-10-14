#include "armor_ekf/pipeline.hpp"
#include <armor_interfaces/msg/yaw_estimate.hpp>
#include <cv_bridge/cv_bridge.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <tf2_ros/static_transform_broadcaster.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/videoio.hpp>
#include <chrono>
#include <filesystem>

namespace armor_ekf {
class ArmorEkfNode : public rclcpp::Node {
public:
  ArmorEkfNode() : Node("armor_ekf") {
    Config c;
    c.reprojection.enabled = declare_parameter("enable_reprojection", c.reprojection.enabled);
    c.reprojection.radius = declare_parameter("robot_radius", c.reprojection.radius);
    c.reprojection.pitch = declare_parameter("model_pitch", c.reprojection.pitch);
    c.reprojection.roll = declare_parameter("model_roll", c.reprojection.roll);
    c.yaw_frame = declare_parameter("yaw_frame", c.yaw_frame);
    c.min_world_horizontal_normal = declare_parameter("min_world_horizontal_normal", c.min_world_horizontal_normal);
    auto frame_array = [this](const char * name, std::array<double, 3> & values) {
      const auto input = declare_parameter<std::vector<double>>(name, std::vector<double>(values.begin(), values.end()));
      if (input.size() != 3) throw std::invalid_argument(std::string(name) + " must have 3 values");
      std::copy(input.begin(), input.end(), values.begin());
    };
    frame_array("odom_gimbal_xyz", c.frames.odom_gimbal_xyz);
    frame_array("odom_gimbal_rpy", c.frames.odom_gimbal_rpy);
    frame_array("gimbal_camera_xyz", c.frames.gimbal_camera_xyz);
    frame_array("gimbal_camera_rpy", c.frames.gimbal_camera_rpy);
    frame_array("camera_optical_xyz", c.frames.camera_optical_xyz);
    frame_array("camera_optical_rpy", c.frames.camera_optical_rpy);
    c.model_path = declare_parameter<std::string>("model_path", "");
    c.confidence = declare_parameter("confidence", c.confidence);
    c.armor_width = declare_parameter("armor_width", c.armor_width);
    c.armor_height = declare_parameter("armor_height", c.armor_height);
    armor_width_ = c.armor_width; armor_height_ = c.armor_height;
    c.max_reprojection_error = declare_parameter("max_reprojection_error", c.max_reprojection_error);
    c.measurement_std = declare_parameter("measurement_std", c.measurement_std);
    c.acceleration_std = declare_parameter("acceleration_std", c.acceleration_std);
    c.reset_timeout = declare_parameter("reset_timeout", c.reset_timeout);
    c.association_gate_scale = declare_parameter("association_gate_scale", c.association_gate_scale);
    c.association_mode = declare_parameter("association_mode", c.association_mode);
    c.target_switch_timeout = declare_parameter("target_switch_timeout", c.target_switch_timeout);
    c.threads = declare_parameter("inference_threads", c.threads);
    const auto corners = declare_parameter<std::vector<int64_t>>("corner_order", {0,1,2,3});
    const auto K = declare_parameter<std::vector<double>>("camera_matrix", std::vector<double>(c.camera_matrix.begin(),c.camera_matrix.end()));
    const auto D = declare_parameter<std::vector<double>>("distortion", std::vector<double>(c.distortion.begin(),c.distortion.end()));
    if (corners.size() != 4 || K.size() != 9 || D.size() != 5) throw std::invalid_argument("Incorrect camera or corner array size");
    std::copy(corners.begin(),corners.end(),c.corner_order.begin());
    std::copy(K.begin(),K.end(),c.camera_matrix.begin()); std::copy(D.begin(),D.end(),c.distortion.begin());
    publish_debug_ = declare_parameter("publish_debug", true);
    pipeline_ = std::make_unique<Pipeline>(c);
    const auto input_topic = declare_parameter<std::string>("image_topic", "/armor/image_raw");
    const auto video_path = declare_parameter<std::string>("video_path", "");
    const auto video_frame_id = declare_parameter<std::string>("frame_id", "camera_optical_frame");
    const auto video_loop = declare_parameter<bool>("loop", false);
    const auto requested_fps = declare_parameter<double>("fps", 0.0);
    world_mode_ = c.yaw_frame == "odom";
    reproject_mode_ = c.reprojection.enabled;
    if (reproject_mode_) {
      poses_ = create_publisher<geometry_msgs::msg::PoseArray>("/armor/reproject/poses", 10);
      projection_markers_ = create_publisher<visualization_msgs::msg::MarkerArray>("/armor/reproject/markers", 10);
    }
    output_ = create_publisher<armor_interfaces::msg::YawEstimate>(world_mode_ ? "/armor/world_yaw" : "/armor/yaw", 10);
    image_output_ = create_publisher<sensor_msgs::msg::Image>(reproject_mode_ ? "/armor/reproject/image_raw" : (world_mode_ ? "/armor/world/image_raw" : "/armor/ekf/image_raw"), rclcpp::QoS(1).reliable());
    if (world_mode_) {
      markers_ = create_publisher<visualization_msgs::msg::MarkerArray>("/armor/world_markers", 10);
      static_tf_ = std::make_unique<tf2_ros::StaticTransformBroadcaster>(this);
      const TransformChain chain(c.frames);
      const std::array<std::string, 4> names{"odom", "gimbal_link", "camera_link", "camera_optical_frame"};
      std::vector<geometry_msgs::msg::TransformStamped> transforms;
      for (size_t i = 0; i < 3; ++i) {
        geometry_msgs::msg::TransformStamped t;
        t.header.stamp = now(); t.header.frame_id = names[i]; t.child_frame_id = names[i+1];
        const auto & edge = chain.edges()[i];
        t.transform.translation.x = edge.translation().x();
        t.transform.translation.y = edge.translation().y();
        t.transform.translation.z = edge.translation().z();
        const Eigen::Quaterniond q(edge.linear());
        t.transform.rotation.x = q.x(); t.transform.rotation.y = q.y();
        t.transform.rotation.z = q.z(); t.transform.rotation.w = q.w();
        transforms.push_back(t);
      }
      static_tf_->sendTransform(transforms);
    }
    if (!video_path.empty()) {
      if (!std::filesystem::is_regular_file(video_path))
        throw std::invalid_argument("video_path does not name a file: " + video_path);
      video_.open(video_path);
      if (!video_.isOpened()) throw std::runtime_error("cannot open video: " + video_path);
      auto fps = requested_fps > 0.0 ? requested_fps : video_.get(cv::CAP_PROP_FPS);
      if (!(fps > 0.0)) fps = 30.0;
      video_loop_ = video_loop;
      frame_id_ = video_frame_id;
      image_input_ = create_publisher<sensor_msgs::msg::Image>(input_topic, rclcpp::QoS(10).reliable());
      video_timer_ = create_wall_timer(std::chrono::duration<double>(1.0 / fps), [this]() { publishVideoFrame(); });
      RCLCPP_INFO(get_logger(), "integrated video source publishing and processing %s on %s at %.2f FPS",
                  video_path.c_str(), input_topic.c_str(), fps);
    } else {
      // Camera/external source mode. Only retain the newest image if inference is slower than the source.
      input_ = create_subscription<sensor_msgs::msg::Image>(input_topic, rclcpp::SensorDataQoS().keep_last(1),
        [this](sensor_msgs::msg::Image::ConstSharedPtr msg) { process(msg); });
    }
    RCLCPP_INFO(get_logger(), "YOLO Pose + IPPE PnP + circular EKF ready; yaw frame: %s; model: %s", c.yaw_frame.c_str(), c.model_path.c_str());
  }
private:
  void publishVideoFrame() {
    cv::Mat frame;
    if (!video_.read(frame) || frame.empty()) {
      if (!video_loop_) {
        RCLCPP_INFO(get_logger(), "video finished");
        video_timer_->cancel();
        return;
      }
      video_.set(cv::CAP_PROP_POS_FRAMES, 0);
      if (!video_.read(frame) || frame.empty()) {
        RCLCPP_ERROR(get_logger(), "failed to read video frame");
        video_timer_->cancel();
        return;
      }
    }
    auto message = cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", frame).toImageMsg();
    message->header.stamp = now();
    message->header.frame_id = frame_id_;
    image_input_->publish(*message);
    // Process the very same frame in this node; no ROS topic round-trip or second inference node is needed.
    process(message);
  }

  void process(const sensor_msgs::msg::Image::ConstSharedPtr & image_msg) {
    try {
      if (image_msg->header.frame_id != "camera_optical_frame")
        throw std::invalid_argument("Image frame_id must be camera_optical_frame");
      const auto image = cv_bridge::toCvShare(image_msg, "bgr8");
      const auto result = pipeline_->process(image->image, rclcpp::Time(image_msg->header.stamp).seconds());
      armor_interfaces::msg::YawEstimate message;
      message.header = image_msg->header;
      message.header.frame_id = result.frame_id;
      message.detection_valid = result.detection_valid; message.measurement_used = result.measurement_used;
      message.filter_initialized = result.filter_initialized; message.target_reset = result.target_reset;
      message.track_id = result.track_id; message.class_id = result.class_id;
      message.visual_track_id = result.visual_track_id; message.target_switched = result.target_switched;
      message.detection_confidence = result.confidence; message.observed_yaw = result.observed_yaw;
      message.filtered_yaw = result.filtered_yaw; message.yaw_rate = result.yaw_rate;
      message.yaw_variance = result.yaw_variance; message.reprojection_error = result.reprojection_error;
      message.pose.orientation.w = 1;
      if (result.detection_valid) {
        message.pose.position.x = result.output_tvec[0]; message.pose.position.y = result.output_tvec[1]; message.pose.position.z = result.output_tvec[2];
        const auto R = rotationFromRvec(result.output_rvec);
        const Eigen::Quaterniond q(R);
        message.pose.orientation.x = q.x(); message.pose.orientation.y = q.y();
        message.pose.orientation.z = q.z(); message.pose.orientation.w = q.w();
        for (size_t i = 0; i < 4; ++i) {
          message.corners[i].x = result.corners[i].x; message.corners[i].y = result.corners[i].y;
        }
      }
      output_->publish(message);
      if (world_mode_) publishMarkers(message);
      if (reproject_mode_) publishProjection(message.header, result.projection);
      if (publish_debug_ && image_output_->get_subscription_count() > 0) {
        const auto annotated = pipeline_->annotate(image->image, result);
        image_output_->publish(*cv_bridge::CvImage(image_msg->header, "bgr8", annotated).toImageMsg());
      }
    } catch (const std::exception & error) {
      RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 2000, "Image processing failed: %s", error.what());
    }
  }
  void publishMarkers(const armor_interfaces::msg::YawEstimate & message) {
    visualization_msgs::msg::MarkerArray array;
    for (int id = 0; id < 2; ++id) {
      visualization_msgs::msg::Marker marker;
      marker.header = message.header; marker.ns = "armor_world"; marker.id = id;
      marker.action = message.detection_valid ? marker.ADD : marker.DELETE;
      marker.pose = message.pose;
      marker.lifetime = rclcpp::Duration::from_seconds(0.3);
      marker.color.a = 1.0f;
      if (id == 0) {
        marker.type = marker.CUBE;
        marker.scale.x = 0.01; marker.scale.y = armor_width_; marker.scale.z = armor_height_;
        marker.color.r = 1.0f; marker.color.g = 0.55f;
      } else {
        marker.type = marker.ARROW;
        if (!message.filter_initialized) marker.action = marker.DELETE;
        marker.pose.orientation.x = marker.pose.orientation.y = 0;
        marker.pose.orientation.z = message.filter_initialized ? std::sin(message.filtered_yaw / 2) : 0;
        marker.pose.orientation.w = message.filter_initialized ? std::cos(message.filtered_yaw / 2) : 1;
        marker.scale.x = 0.25; marker.scale.y = 0.02; marker.scale.z = 0.03;
        marker.color.g = 1.0f;
      }
      array.markers.push_back(marker);
    }
    markers_->publish(array);
  }
  void publishProjection(const std_msgs::msg::Header & header, const RobotProjection & projection) {
    geometry_msgs::msg::PoseArray poses; poses.header = header;
    visualization_msgs::msg::MarkerArray markers;
    for (int i=0;i<5;++i) {
      visualization_msgs::msg::Marker m;
      m.header=header; m.ns="four_armor_model"; m.id=i;
      m.action=projection.valid ? m.ADD : m.DELETE;
      m.lifetime=rclcpp::Duration::from_seconds(0.3);
      m.color.a=1; m.pose.orientation.w=1;
      if (projection.valid) {
        if (i<4) {
          const auto & p=projection.plates[i].world_pose;
          m.pose.position.x=p.translation().x(); m.pose.position.y=p.translation().y(); m.pose.position.z=p.translation().z();
          const Eigen::Quaterniond q(p.linear());
          m.pose.orientation.x=q.x(); m.pose.orientation.y=q.y(); m.pose.orientation.z=q.z(); m.pose.orientation.w=q.w();
          poses.poses.push_back(m.pose);
        } else {
          m.pose.position.x=projection.center.x(); m.pose.position.y=projection.center.y(); m.pose.position.z=projection.center.z();
        }
      }
      m.type=i<4 ? m.CUBE : m.SPHERE;
      m.scale.x=i<4 ? 0.01 : 0.04; m.scale.y=i<4 ? armor_width_ : 0.04; m.scale.z=i<4 ? armor_height_ : 0.04;
      if (i==0) { m.color.r=1; m.color.g=0.7f; }
      else if (i<4) { m.color.g=0.85f; m.color.b=1; }
      else m.color.r=1;
      markers.markers.push_back(m);
    }
    poses_->publish(poses); projection_markers_->publish(markers);
  }
  std::unique_ptr<Pipeline> pipeline_;
  cv::VideoCapture video_;
  bool video_loop_ = false;
  std::string frame_id_ = "camera_optical_frame";
  rclcpp::TimerBase::SharedPtr video_timer_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr image_input_;
  bool publish_debug_;
  bool world_mode_ = false, reproject_mode_ = false;
  rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr poses_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr projection_markers_;
  double armor_width_ = 0.135, armor_height_ = 0.055;
  std::unique_ptr<tf2_ros::StaticTransformBroadcaster> static_tf_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr markers_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr input_;
  rclcpp::Publisher<armor_interfaces::msg::YawEstimate>::SharedPtr output_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr image_output_;
};
} // namespace armor_ekf

int main(int argc, char ** argv) {
  rclcpp::init(argc, argv);
  try { rclcpp::spin(std::make_shared<armor_ekf::ArmorEkfNode>()); }
  catch (const std::exception & error) {
    std::cerr << "armor_ekf: " << error.what() << '\n';
    rclcpp::shutdown(); return 1;
  }
  rclcpp::shutdown(); return 0;
}
