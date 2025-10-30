#ifndef VISUAL_ODOMETRY_HPP_
#define VISUAL_ODOMETRY_HPP_

#include "slam_core/common_include.hpp"
#include "slam_core/frame.hpp"
#include "slam_core/G2oTypes.hpp"

class VisualOdometry
{
    public:
        VisualOdometry(cv::Mat &K);

        //slam_system에서 visual odometry를 위해서 호출하는 메소드
        void visual_odometry(cv::Mat &img_1, cv::Mat &img_2);

        bool triangulation(const Frame &frame_1, Frame &frame_2, 
                            const std::vector<cv::DMatch> &matches, std::vector<std::shared_ptr<MapPoint>> &points, bool isFirst = false);
        
        bool pose_estimate_2d2d(const Frame &frame_1, const Frame &frame_2, const std::vector<cv::DMatch> &matches);

        bool PnPcompute(const VecVector3d &points_3d, const VecVector2d &points_2d, Frame &cur_frame);

        bool PnPcompute_g2o(const VecVector3d &points_3d, const VecVector2d &points_2d, Frame &cur_frame);

        bool check_parrallax(const Frame &frame_1, const std::vector<cv::KeyPoint> &kp2, const std::vector<cv::DMatch> &matches, double min_parallax_deg);
        
        const std::vector<uchar>& pose_inlier_mask() const { return pose_inlier_mask_; }

        Sophus::SE3d get_rel_pose();

    private:

        float calc_homography_score(const std::vector<cv::Point2f> &points1, const std::vector<cv::Point2f> &points2, cv::Mat &H_21);
            
        float calc_fundamental_score(const std::vector<cv::Point2f> &points1, const std::vector<cv::Point2f> &points2, cv::Mat &F_21);

        // bool rescaleInitialMap(std::vector<Vec3> &world_points, Sophus::SE3d &T_21);

        bool rescaleInitialMap(std::vector<cv::Point3d> &world_points, Sophus::SE3d &T_21);

    private:
        cv::Mat R, t, K;
        Sophus::SE3d T_;
        std::vector<uchar> pose_inlier_mask_;
};

#endif
