#ifndef FRAME_HPP_
#define FRAME_HPP_

#include "slam_core/common_include.hpp"
#include "mappoint.hpp"

class Frame {
    public:
        unsigned long id_ = 0;
        unsigned long keyframe_id_ = 0;
        bool is_keyframe_ = false;
        Sophus::SE3d pose_;
        cv::Mat img_;

        std::vector<cv::KeyPoint> keypoints_;
        cv::Mat descriptors_;
        std::vector<std::shared_ptr<MapPoint>> observed_map_points_;


    public:
        Frame() {}

        Frame(long id, const Sophus::SE3d &pose, const cv::Mat &img): id_(id), pose_(pose), img_(img) {}

        Sophus::SE3d get_pose() const{
            return pose_;
        }

        void set_pose(const Sophus::SE3d &pose){
            pose_ = pose;
        }

        void set_keyframe(unsigned long keyframe_id) {
            this->is_keyframe_ = true;
            this->keyframe_id_ = keyframe_id;
        }

        static std::shared_ptr<Frame> CreateFrame();
};

#endif