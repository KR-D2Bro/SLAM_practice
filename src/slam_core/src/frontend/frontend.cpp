#include "slam_core/frontend.hpp"

using namespace std;
using namespace cv;

Frontend::Frontend(std::shared_ptr<vector<shared_ptr<MapPoint>>> map_points) : map_points_(map_points) {
    feature_tracker_ = std::make_unique<FeatureTracker>(K);
    visual_odometry_ = std::make_unique<VisualOdometry>(K);
    cur_frame = std::make_shared<Frame>(-1, Sophus::SE3d(), cv::Mat()); prev_frame = nullptr;
}

//0: 실패, 1: 일반 프레임, 2: 키프레임
int8_t Frontend::run(cv::Mat &img){
    prev_frame = cur_frame;

    cur_frame = std::make_shared<Frame>(frame_cnt++, prev_frame->get_pose(), feature_tracker_->undistort(img));
    feature_tracker_->detectAndCompute(*cur_frame);
    
    //첫번 째 frame을 key frame으로 등록 후 종료: 비교할 이미지 없음.
    if(keyframes.size()==0){
        return add_keyframe(cur_frame);
    }
    else if(keyframes.size() == 1){ //아직 3D 포인트가 없으므로 2d2d pose estimation
        //새로운 키프레임 추가
        vector<DMatch> matches;
        feature_tracker_->track_feature(*last_keyframe, *cur_frame, matches);
        if(visual_odometry_->pose_estimate_2d2d(*last_keyframe, *cur_frame, matches)){
            Sophus::SE3d rel_pose = visual_odometry_->get_rel_pose();
            cur_frame->set_pose(last_keyframe->get_pose() * rel_pose);
            cout << "cur_frame pose: \n" << cur_frame->get_pose().matrix() << endl;

            // triangulation 함수 호출
            if (visual_odometry_->triangulation(*last_keyframe, *cur_frame,
                                                matches,
                                                *map_points_, true))
            {      
                return add_keyframe(cur_frame);
            }
            return 0;
        }
        return 0;
    }
    else{   
        VecVector3d points_3d;
        VecVector2d points_2d;
        vector<DMatch> matches;
        if(feature_tracker_->match_3d_2d(*map_points_, *cur_frame, points_3d, points_2d, matches)){
            //Pnp 수행.
            if(visual_odometry_->PnPcompute_g2o(points_3d, points_2d, *cur_frame)){
                // 기본 키프레임 생성 조건
                // 1. 30프레임 이상 간격
                const bool c1 = frame_cnt >= last_keyframe->id_ + 30; 

                // 2. 트래킹 포인트가 키프레임 관측 포인트의 70% 이하이면서 15개 이상
                // PnPcompute_g2o에서 pose_inlier_mask_ 설정됨
                vector<uchar> pose_inlier_mask = visual_odometry_->pose_inlier_mask();
                int num_inliers = std::count(pose_inlier_mask.begin(), pose_inlier_mask.end(), 1);
                const bool c2 = num_inliers < last_keyframe->observed_map_points_.size() * 0.7 && num_inliers > 15;

                // 3. 이동량 검사
                Sophus::SE3d rel = last_keyframe->get_pose().inverse() * cur_frame->get_pose();
                double trans = rel.translation().norm();
                double rot = Sophus::SO3d(rel.rotationMatrix()).log().norm(); // 라디안
                const double min_trans = 0.05;     // 데이터에 맞게 조정
                const double min_rot = 3.0 * M_PI / 180.0;
                bool c_motion = (trans > min_trans) || (rot > min_rot);
                // c_motion = true; // 일단 모션 조건 무시

                // 4. 시차 검사
                vector<DMatch> dummy_matches;
                feature_tracker_->track_feature(*last_keyframe, *cur_frame, dummy_matches);
                // 시차 계산시 pose_inlier_mask_를 이용하여 inlier 매칭점만 사용하도록 수정
                bool c_parrallax = false;
                c_parrallax = visual_odometry_->check_parrallax(*last_keyframe, *cur_frame, dummy_matches, 1.2);

                cout << "KeyFrame conditions: " << c1 << ", " << c2 << ", " << c_motion << ", " << c_parrallax << endl;

                if((c1 || c2) && c_motion && c_parrallax){
                    //트래킹 포인트가 너무 적으면 키프레임 추가
                    cout << "Adding new KeyFrame." << endl;
                    // triangulation 함수 호출
                    if (visual_odometry_->triangulation(*last_keyframe, *cur_frame,
                                                        matches, *map_points_))
                    {
                        return add_keyframe(cur_frame);
                    }
                }
                return 1;
            }
            return 0;
        }
        return 0;
    }
}

int Frontend::add_keyframe(std::shared_ptr<Frame> &frame){
    last_keyframe = frame;
    frame->set_keyframe(static_cast<unsigned long>(keyframes.size()));
    keyframes.push_back(frame);
    cout << "key frame added" << endl;
    
    return 2;
}
