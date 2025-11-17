#include "slam_core/frontend.hpp"

#include "slam_core/mappoint.hpp"

using namespace std;
using namespace cv;

Frontend::Frontend(std::shared_ptr<vector<shared_ptr<MapPoint>>> map_points) : map_points_(map_points) {
    feature_tracker_ = std::make_unique<FeatureTracker>(K);
    visual_odometry_ = std::make_unique<VisualOdometry>(K);
    opticalflow_tracker_ = std::make_unique<OpticalFlowTracker>();
    cur_frame = std::make_shared<Frame>(-1, Sophus::SE3d(), cv::Mat()); prev_frame = nullptr;
}

//0: 실패, 1: 일반 프레임, 2: 키프레임
int8_t Frontend::run(cv::Mat &img){
    Sophus::SE3d initial_rel_pose = prev_frame ? prev_frame->get_pose().inverse() * cur_frame->get_pose() : Sophus::SE3d();

    prev_frame = cur_frame;

    cur_frame = std::make_shared<Frame>(frame_cnt++, prev_frame->get_pose(), feature_tracker_->undistort(img));
    feature_tracker_->detectAndCompute(*cur_frame);

    // if(prev_frame->id_ != -1){
    //     // 매 프레임 Optical flow 트래킹 수행.
    //     vector<KeyPoint> kp2;
    //     vector<bool> success;
    //     opticalflow_tracker_->track_pyramid_opticalflow(*prev_frame, *cur_frame, kp2, success, true);
    //     per_frame_parallax = cal_parallax_opticalflow(*prev_frame, kp2, success);
    // }
    
    
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
                return add_keyframe(cur_frame, map_points_->size());
            }
            return 0;
        }
        return 0;
    }
    else{   
        VecVector3d points_3d;
        VecVector2d points_2d;
        vector<DMatch> matches;
        vector<shared_ptr<MapPoint>> inliers_mappoints;
        if(feature_tracker_->match_3d_2d(*map_points_, *cur_frame, points_3d, points_2d, matches, inliers_mappoints)){
            //Pnp 수행.
            if(visual_odometry_->PnPcompute_g2o(points_3d, points_2d, *cur_frame)){
                // inliers_mappoints의 디스크립터 업데이트 및 observed_map_points_ 설정
                vector<uchar> pose_inlier_mask = visual_odometry_->pose_inlier_mask();
                for(int i=0; i<matches.size(); i++){
                    auto &m = matches[i];
                    if(pose_inlier_mask[i]){
                        inliers_mappoints[i]->update_descriptor(cur_frame->descriptors_.row(m.trainIdx));
                        cur_frame->observed_map_points_[m.trainIdx] = inliers_mappoints[i];
                    }
                }

                // 기본 키프레임 생성 조건
                // 1. 30프레임 이상 간격
                const bool c1 = frame_cnt >= last_keyframe->id_ + 25; 

                // 2. 트래킹 포인트가 키프레임 관측 포인트의 70% 이하이면서 15개 이상
                // PnPcompute_g2o에서 pose_inlier_mask_ 설정됨
                int num_inliers = std::count(pose_inlier_mask.begin(), pose_inlier_mask.end(), 1);
                const bool c2 = num_inliers < last_points_num * 0.7 && num_inliers > 15;
                cout << "PnP inliers: " << num_inliers << " / " << last_points_num << endl;

                // 3. 이동량 검사
                Sophus::SE3d rel = last_keyframe->get_pose().inverse() * cur_frame->get_pose();
                double trans = rel.translation().norm();
                const double min_trans = 0.1;     // 데이터에 맞게 조정
                bool c_motion = (trans > min_trans);

                // 4. 시차 검사
                total_parallax += per_frame_parallax;
                bool c_parrallax = total_parallax > 5.0;
                // c_parrallax = true;
                // c_parrallax = visual_odometry_->check_parrallax(*last_keyframe, dummy_kp2, inliers_matches, 5.0);

                cout << "KeyFrame conditions: " << c1 << ", " << c2 << ", " << c_motion << ", " << c_parrallax << endl;

                if(c1 || (c_motion && c_parrallax)){
                    //트래킹 포인트가 너무 적으면 키프레임 추가
                    cout << "Adding new KeyFrame." << endl;
                    // triangulation 함수 호출
                    vector<DMatch> dummy_matches;
                    feature_tracker_->track_feature(*last_keyframe, *cur_frame, dummy_matches);
                    if (visual_odometry_->triangulation(*last_keyframe, *cur_frame,
                                                    dummy_matches, *map_points_))
                    {
                        return add_keyframe(cur_frame, dummy_matches.size());
                    }
                }
                return 1;
            }
            return 0;
        }
        return 0;
    }
}

int Frontend::add_keyframe(std::shared_ptr<Frame> &frame, int num_inliers){
    last_keyframe = frame;
    frame->set_keyframe(static_cast<unsigned long>(keyframes.size()));
    keyframes.push_back(frame);
    cout << "key frame added" << endl;

    last_points_num = num_inliers;
    total_parallax = 0.0;
    
    return 2;
}

// Optical flow로 얻은 키포인트 매칭으로 시차 계산
double Frontend::cal_parallax_opticalflow(const Frame &frame_1, const vector<KeyPoint> &kp2, const vector<bool> &success){
    vector<DMatch> inliers_matches, optical_matches;

    for(int i = 0; i<(int)success.size(); i++){
        cv::DMatch m(i, i, 0.0f);
        optical_matches.push_back(m);
    }

    for(size_t i = 0; i < optical_matches.size(); i++){
        if(success[i] == true){
            inliers_matches.push_back(optical_matches[i]);
        }
    }
    return visual_odometry_->check_parrallax(frame_1, kp2, inliers_matches);
}