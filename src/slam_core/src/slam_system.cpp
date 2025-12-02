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

struct GroundTruthPose {
    long long timestamp; // 나노초 단위
    double x, y, z;      // 위치
    double qw, qx, qy, qz; // 회전 (Quaternion)
};

void publishGTPause(rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub);
// CSV 파일을 읽는 함수
std::vector<GroundTruthPose> loadEuRoCGroundTruth(const std::string& filename);
Sophus::SE3d groundTruthToSE3(const GroundTruthPose& pose);

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
            gt_traj_publisher_ = this->create_publisher<nav_msgs::msg::Path>("/gt_trajectory", 10);

            publishGTPause(gt_traj_publisher_);
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
        std::unique_ptr<Frontend> frontend_;
        std::unique_ptr<LocalMapping> local_mapping_;
        rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_subscription_;
        rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pose_publisher_;
        rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr keyframe_pose_publisher_, gt_traj_publisher_;
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


void publishGTPause(rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub) {
    auto gt_data = loadEuRoCGroundTruth("/home/dongjae/V1_01_easy/data.csv");

    if (gt_data.empty()) {
        return;
    }
    // EuRoC cam0: body(IMU)->cam extrinsic
    Eigen::Matrix3d R_bc;
    R_bc << 0.0148655429818, -0.999880929698, 0.00414029679422,
            0.999557249008,  0.0149672133247, 0.025715529948,
           -0.0257744366974, 0.00375618835797, 0.999660727178;
    const Eigen::Vector3d t_bc(-0.0216401454975, -0.064676986768, 0.00981073058949);
    const Sophus::SE3d T_bc(R_bc, t_bc);

    // 기준 포즈: GT 첫 바디포즈 -> 카메라 -> ROS 로 변환 후 역행렬
    const Sophus::SE3d T_wb0 = groundTruthToSE3(gt_data.front());
    const Sophus::SE3d T_wc0 = T_wb0 * T_bc;
    const Sophus::SE3d T_ros0 = convert_cv_to_ros(T_wc0);
    const Sophus::SE3d T_ros0_inv = T_ros0.inverse();
    
    nav_msgs::msg::Path path_msg;
    path_msg.header.frame_id = "map"; // 혹은 "world"
    path_msg.header.stamp = rclcpp::Clock().now();

    for (const auto& pose : gt_data) {
        const Sophus::SE3d T_wb = groundTruthToSE3(pose);
        const Sophus::SE3d T_wc = T_wb * T_bc;
        const Sophus::SE3d T_ros = convert_cv_to_ros(T_wc);
        const Sophus::SE3d T_rel = T_ros0_inv * T_ros; // 시작을 원점/단위회전에 정렬

        geometry_msgs::msg::PoseStamped ps;
        ps.header = path_msg.header; 
        ps.pose.position.x = T_rel.translation().x();
        ps.pose.position.y = T_rel.translation().y();
        ps.pose.position.z = T_rel.translation().z();
        ps.pose.orientation.w = T_rel.unit_quaternion().w();
        ps.pose.orientation.x = T_rel.unit_quaternion().x();
        ps.pose.orientation.y = T_rel.unit_quaternion().y();
        ps.pose.orientation.z = T_rel.unit_quaternion().z();
        path_msg.poses.push_back(ps);
    }

    path_pub->publish(path_msg);
}

// CSV 파일을 읽는 함수
std::vector<GroundTruthPose> loadEuRoCGroundTruth(const std::string& filename) {
    std::vector<GroundTruthPose> gt_poses;
    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cerr << "파일을 열 수 없습니다: " << filename << std::endl;
        return gt_poses;
    }

    std::string line;
    // 첫 번째 줄(헤더)은 건너뜁니다 (#timestamp, p_RS_R_x, ...)
    std::getline(file, line);

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string cell;
        std::vector<std::string> row;

        // 콤마(,)를 기준으로 분리
        while (std::getline(ss, cell, ',')) {
            row.push_back(cell);
        }

        // 데이터 파싱 (이미지의 컬럼 순서에 맞춤)
        // 0: timestamp, 1: px, 2: py, 3: pz, 4: qw, 5: qx, 6: qy, 7: qz
        if (row.size() >= 8) {
            GroundTruthPose pose;
            pose.timestamp = std::stoll(row[0]);
            pose.x = std::stod(row[1]);
            pose.y = std::stod(row[2]);
            pose.z = std::stod(row[3]);
            pose.qw = std::stod(row[4]);
            pose.qx = std::stod(row[5]);
            pose.qy = std::stod(row[6]);
            pose.qz = std::stod(row[7]);
            
            gt_poses.push_back(pose);
        }
    }
    return gt_poses;
}

Sophus::SE3d groundTruthToSE3(const GroundTruthPose& pose) {
    Eigen::Quaterniond q(pose.qw, pose.qx, pose.qy, pose.qz);
    q.normalize();
    const Eigen::Vector3d t(pose.x, pose.y, pose.z);
    return Sophus::SE3d(q, t);
}
