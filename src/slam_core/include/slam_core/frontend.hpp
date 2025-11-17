#ifndef FRONTEND_HPP_
#define FRONTEND_HPP_

#include "slam_core/common_include.hpp"
#include "slam_core/feature_tracker.hpp"
#include "slam_core/visual_odometry.hpp"
#include "slam_core/opticalflow_tracker.hpp"

struct MapPoint;

class Frontend
{
    public:
        Frontend(std::shared_ptr<std::vector<std::shared_ptr<MapPoint>>> map_points);

        std::int8_t run(cv::Mat &img);

        std::shared_ptr<Frame> cur_frame;
        std::vector<std::shared_ptr<Frame>> keyframes;
        std::vector<std::shared_ptr<Frame>> frames;

    private:
        double cal_parallax_opticalflow(const Frame &frame_1, const std::vector<cv::KeyPoint> &kp2, const std::vector<bool> &success);

        std::unique_ptr<FeatureTracker> feature_tracker_;
        std::unique_ptr<VisualOdometry> visual_odometry_;
        std::unique_ptr<OpticalFlowTracker> opticalflow_tracker_;
        std::shared_ptr<Frame> prev_frame, last_keyframe;
        std::shared_ptr<std::vector<std::shared_ptr<MapPoint>>> map_points_;

        unsigned long frame_cnt = 0;
        unsigned long last_points_num = 0;
        double per_frame_parallax = 0.0, total_parallax = 0.0;
        cv::Mat K = (cv::Mat_<double>(3,3) << 458.654, 0, 367.215, 0, 457.296, 248.375, 0, 0, 1); 

        int add_keyframe(std::shared_ptr<Frame> &cur_frame, int num_inliers=0);
};

#endif
