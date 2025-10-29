#ifndef MAPPOINT_HPP_
#define MAPPOINT_HPP_

#include "slam_core/common_include.hpp"
#include <list>
#include <memory>

class Frame;

struct MapPoint {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        
        typedef std::shared_ptr<MapPoint> Ptr;

        unsigned long id_ = 0;
        bool is_outlier_ = false;
        Vec3 pos_ = Vec3::Zero();   //position in world coordinates
        // cv::Point3d color_;

        int observed_cnt_ = 0;
        cv::Mat descriptor_; //descriptor for matching
        std::list<std::weak_ptr<Frame>> observations_;

    public:
        MapPoint() {}

        MapPoint(long id, Vec3 position, cv::Mat descriptor) : id_(id), pos_(position), descriptor_(descriptor) {}

        Vec3 get_pos() const{
            return pos_;
        }

        void set_pos(const Vec3 &pos){
            pos_ = pos;
        }

        void add_observation(std::shared_ptr<Frame> frame){
            observations_.push_back(frame);
            observed_cnt_++;
        }

        void remove_observation(std::shared_ptr<Frame> frame){
            observations_.remove_if([frame](const std::weak_ptr<Frame>& wp) {
                auto sp = wp.lock();
                return sp && sp->id_ == frame->id_;
            });
            observed_cnt_ = static_cast<int>(observations_.size());
        }

        static MapPoint::Ptr CreateNewMappoint(long id, Vec3 position, cv::Mat descriptor){
            return std::make_shared<MapPoint>(id, position, descriptor);
        }
};

#endif
