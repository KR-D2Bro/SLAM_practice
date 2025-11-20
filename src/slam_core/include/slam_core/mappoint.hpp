#ifndef MAPPOINT_HPP_
#define MAPPOINT_HPP_

#include "slam_core/common_include.hpp"
#include <list>
#include <memory>
#include <unordered_map>

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
        std::unordered_map<Frame*, int> observations_; // 관측 프레임과 키포인트 인덱스 매핑

    public:
        MapPoint() {}

        MapPoint(long id, Vec3 position, cv::Mat descriptor) : id_(id), pos_(position), descriptor_(descriptor) {}

        Vec3 get_pos() const{
            return pos_;
        }

        void set_pos(const Vec3 &pos){
            pos_ = pos;
        }

        void add_observation(const std::shared_ptr<Frame> &frame, int keypoint_index){
            observations_.emplace(frame.get(), keypoint_index);
            observed_cnt_++;
        }

        void remove_observation(std::shared_ptr<Frame> frame);

        static MapPoint::Ptr CreateNewMappoint(long id, Vec3 position, cv::Mat descriptor){
            return std::make_shared<MapPoint>(id, position, descriptor);
        }

        inline void update_descriptor(const cv::Mat &new_descriptor){
            descriptor_ = new_descriptor.clone();
        }
};

#endif
