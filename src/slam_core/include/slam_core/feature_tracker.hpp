#ifndef FEATURE_TRACKER_HPP_
#define FEATURE_TRACKER_HPP_

#include<slam_core/common_include.hpp>
#include "slam_core/frame.hpp"

typedef std::vector<std::uint32_t> DescType;

class FeatureTracker
{
    public:
        FeatureTracker(cv::Mat &K);

        void detectAndCompute(Frame &frame);
        
        void track_feature(const Frame &frame_1, const Frame &frame_2, std::vector<cv::DMatch> &matches, float ratio = 0.7, bool visualize = true); 
        
        cv::Mat undistort(cv::Mat &img);

        //inliers_mappoints: 2d 키포인트와 매칭된 3D 포인트들
        bool match_3d_2d(const std::vector<std::shared_ptr<MapPoint>> &map_points, const Frame &cur_frame, 
            VecVector3d &points_3d, VecVector2d &points_2d, std::vector<cv::DMatch> &matches, std::vector<std::shared_ptr<MapPoint>> &inliers_mappoints);

        // std::vector<cv::DMatch> matches;
    private:
        void ComputeORB(cv::Mat &img, std::vector<cv::KeyPoint> &key_points, std::vector<DescType> &descriptors);
        void BfMatch(const cv::Mat &desc1, const cv::Mat &desc_2, std::vector<cv::DMatch> &matches, float ratio = 0.7);
        cv::DMatch queryMatch(const Frame &frame, const Vec2 &mp2pix, const cv::Mat &mpDescriptor, std::vector<std::vector<int>> &cells, int cols, int rows, float search_area = 24.0f);
        

        cv::Ptr<cv::FeatureDetector> detector_;
        cv::Ptr<cv::DescriptorExtractor> descriptor_;
        cv::Ptr<cv::DescriptorMatcher> matcher_;

        cv::Mat K;
};

#endif