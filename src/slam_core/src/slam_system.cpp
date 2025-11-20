#include<cv_bridge/cv_bridge.h>
#include "slam_core/common_include.hpp"
#include "slam_core/frame.hpp"
#include "slam_core/mappoint.hpp"
#include "slam_core/frontend.hpp"
#include "slam_core/Map.hpp"
#include "slam_core/LocalMapping.hpp"
#include<iostream>
#include <chrono>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "slam_core/visual_odometry.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "nav_msgs/msg/path.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "sensor_msgs/point_cloud2_iterator.hpp"

using namespace std;
using namespace cv;

class SlamSystem : public rclcpp::Node
{
    public:
        SlamSystem() : Node("slam_system") {
            image_subscription_ = this->create_subscription<sensor_msgs::msg::Image>
            ("/cam0/image_raw", 10, bind(&SlamSystem::topic_callback, this, placeholders::_1));

            map_ = std::make_shared<Map>();
            frontend_ = std::make_unique<Frontend>(map_);
            cv::Mat K = (cv::Mat_<double>(3,3) << 458.654, 0, 367.215, 0, 457.296, 248.375, 0, 0, 1);
            local_mapping_ = std::make_unique<LocalMapping>(map_, K);
            keyframes_ = std::make_unique<nav_msgs::msg::Path>();

            pose_publisher_ = this->create_publisher<geometry_msgs::msg::PoseStamped>("/pose", 10);
            keyframe_pose_publisher_ =
                this->create_publisher<nav_msgs::msg::Path>("/keyframe_trajectory", 10);
            map_points_publisher_ =
                this->create_publisher<sensor_msgs::msg::PointCloud2>("/map_points", 10);
        }

        ~SlamSystem() {
            local_mapping_.reset();
            frontend_.reset();
            map_.reset();
        }

    private:
        void topic_callback(const sensor_msgs::msg::Image::SharedPtr msg){
            Mat img = cv_bridge::toCvShare(msg,sensor_msgs::image_encodings::MONO8)->image;

            int8_t res = frontend_->run(img); 

            publish_pose(frontend_->cur_frame->get_pose());

            if (res == 2) {
                local_mapping_->insert_kf2queue(frontend_->cur_frame);
                publish_path(keyframe_pose_publisher_);
                publish_map_points();
            }
            
        }

        Sophus::SE3d convert_cv_to_ros(const Sophus::SE3d &pose_cv) const {
            const Sophus::SO3d R_ros_cv_so3(R_ros_cv);
            const Sophus::SO3d R_cv_ros_so3(R_ros_cv.transpose());
            Sophus::SO3d R_ros = R_ros_cv_so3 * pose_cv.so3() * R_cv_ros_so3;
            Eigen::Vector3d t_ros = R_ros_cv * pose_cv.translation();
            return {R_ros, t_ros};
        }

        void publish_path(rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr publisher) {
            nav_msgs::msg::Path path;
            for(const auto &kf : map_->keyframes()){
                Sophus::SE3d pose = kf->get_pose();
                Sophus::SE3d ros_pose = convert_cv_to_ros(pose);
            
                auto pose_stamp = geometry_msgs::msg::PoseStamped();
                pose_stamp.pose.position.x = ros_pose.translation().x();
                pose_stamp.pose.position.y = ros_pose.translation().y();
                pose_stamp.pose.position.z = ros_pose.translation().z();
                pose_stamp.pose.orientation.w = ros_pose.unit_quaternion().w();
                pose_stamp.pose.orientation.x = ros_pose.unit_quaternion().x();
                pose_stamp.pose.orientation.y = ros_pose.unit_quaternion().y();
                pose_stamp.pose.orientation.z = ros_pose.unit_quaternion().z();
                pose_stamp.header.set__stamp(this->get_clock()->now());
                pose_stamp.header.set__frame_id("map");

                path.poses.push_back(pose_stamp);
                path.header.stamp = this->get_clock()->now();
                path.header.frame_id = "map";
            }
            
            publisher->publish(path);
            RCLCPP_INFO_STREAM(this->get_logger(), "Publishing" );
        }

        void publish_pose(const Sophus::SE3d &pose) {
            Sophus::SE3d ros_pose = convert_cv_to_ros(pose);

            auto pose_stamp = geometry_msgs::msg::PoseStamped();
            pose_stamp.pose.position.x = ros_pose.translation().x();
            pose_stamp.pose.position.y = ros_pose.translation().y();
            pose_stamp.pose.position.z = ros_pose.translation().z();
            pose_stamp.pose.orientation.w = ros_pose.unit_quaternion().w();
            pose_stamp.pose.orientation.x = ros_pose.unit_quaternion().x();
            pose_stamp.pose.orientation.y = ros_pose.unit_quaternion().y();
            pose_stamp.pose.orientation.z = ros_pose.unit_quaternion().z();
            pose_stamp.header.set__stamp(this->get_clock()->now());
            pose_stamp.header.set__frame_id("map");

            pose_publisher_->publish(pose_stamp);
        }


        void publish_map_points() {
            const vector<shared_ptr<MapPoint>> &map_points = map_->mappoints();
            if (!map_points_publisher_ || ! map_points.size()) {
                return;
            }

            const auto now = this->get_clock()->now();
            std::size_t valid_count = 0;
            for (const auto &mp : map_points) {
                if (mp && !mp->is_outlier_) {
                    ++valid_count;
                }
            }

            sensor_msgs::msg::PointCloud2 cloud;
            cloud.header.frame_id = "map";
            cloud.header.stamp = now;

            sensor_msgs::PointCloud2Modifier modifier(cloud);
            modifier.setPointCloud2FieldsByString(1, "xyz");
            modifier.resize(valid_count);

            sensor_msgs::PointCloud2Iterator<float> iter_x(cloud, "x");
            sensor_msgs::PointCloud2Iterator<float> iter_y(cloud, "y");
            sensor_msgs::PointCloud2Iterator<float> iter_z(cloud, "z");

            for (const auto &mp : map_points) {
                if (!mp || mp->is_outlier_) {
                    continue;
                }
                const auto pos = mp->get_pos();
                const Eigen::Vector3d pos_ros = R_ros_cv * pos;
                *iter_x = static_cast<float>(pos_ros.x());
                *iter_y = static_cast<float>(pos_ros.y());
                *iter_z = static_cast<float>(pos_ros.z());
                ++iter_x;
                ++iter_y;
                ++iter_z;
            }

            map_points_publisher_->publish(cloud);
        }

        std::shared_ptr<Map> map_;
        cv::Point3f position_;
        std::unique_ptr<nav_msgs::msg::Path> path_, keyframes_;
        std::unique_ptr<Frontend> frontend_;\
        std::unique_ptr<LocalMapping> local_mapping_;
        rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_subscription_;
        rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pose_publisher_;
        rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr keyframe_pose_publisher_;
        rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr map_points_publisher_;
        rclcpp::TimerBase::SharedPtr path_timer_;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<SlamSystem>()); 
    rclcpp::shutdown();
    return 0;
}
