#ifndef MAPPOINT_HPP_
#define MAPPOINT_HPP_

#include "slam_core/common_include.hpp"

struct MapPoint {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        
        typedef std::shared_ptr<MapPoint> Ptr;

        unsigned long id_ = 0;
        bool is_outlier_ = false;
        Vec3 pos_ = Vec3::Zero();   //position in world coordinates
        cv::Point3d color_;

        int observed_cnt_ = 0;
        cv::Mat descriptor_; //descriptor for matching
        // std::list<std::weak_ptr<Feature>> observations_;

    public:
        MapPoint() {}

        MapPoint(long id, Vec3 position, cv::Mat descriptor) : id_(id), pos_(position), descriptor_(descriptor) {}

        Vec3 get_pos() const{
            return pos_;
        }

        void set_pos(const Vec3 &pos){
            pos_ = pos;
        }

        // void add_observation(std::shared_ptr<Feature> feature){
        //     observations_.push_back(feature);
        //     observed_cnt_++;
        // }

        // void remove_observation(std::shared_ptr<Feature> feature);

        static MapPoint::Ptr CreateNewMappoint();
};

#endif