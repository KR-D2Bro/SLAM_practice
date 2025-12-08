#include "slam_core/frontend.hpp"

#include "slam_core/Map.hpp"
#include "slam_core/mappoint.hpp"
#include <limits>


using namespace std;
using namespace cv;

Frontend::Frontend(shared_ptr<Map> &map) : map_(map) {
    feature_tracker_ = std::make_unique<FeatureTracker>(K);
    visual_odometry_ = std::make_unique<VisualOdometry>(K);
    opticalflow_tracker_ = std::make_unique<OpticalFlowTracker>();
    cur_frame = std::make_shared<Frame>(-1, Sophus::SE3d(), cv::Mat()); prev_frame = nullptr;
}

//-1: 실패, 0: 일반 키프레임, 1: 첫 키프레임, 2: 두번째 키프레임, 3: 일반 프레임
int8_t Frontend::run(cv::Mat &img){
    vector<KeyPoint> kp2;
    vector<Point2f> prev_pts2d, cur_pts2d;
    vector<bool> success;
    vector<uchar> status;

    // 이전 프레임 저장 및 현재 프레임 생성
    prev_frame = cur_frame;
    cur_frame = std::make_shared<Frame>(frame_cnt++, prev_frame->get_pose(), feature_tracker_->undistort(img));
    feature_tracker_->detectAndCompute(*cur_frame); // cur_frame의 keypoints_, descriptors_ 세팅

    if(prev_frame->id_ != -1){
        // 매 프레임 Optical flow 트래킹 수행.
        chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
        // opticalflow_tracker_->track_pyramid_opticalflow(*prev_frame, *cur_frame, kp2, success, false);
        KeyPoint::convert(prev_frame->keypoints_,prev_pts2d);
        kp2.resize(prev_pts2d.size());
        cur_pts2d.resize(prev_pts2d.size());
        status.resize(prev_pts2d.size());
        calcOpticalFlowPyrLK(prev_frame->img_, cur_frame->img_, prev_pts2d, cur_pts2d, status, noArray());
        Mat img2_single;
        cv::cvtColor(cur_frame->img_, img2_single, cv::COLOR_GRAY2BGR);
        
        for (int i = 0; i < status.size(); i++) {
            if (status.at(i)) {
                cv::circle(img2_single, cur_pts2d[i], 2, cv::Scalar(0, 250, 0), 2);
                cv::line(img2_single, prev_pts2d[i], cur_pts2d[i], cv::Scalar(0, 250, 0));
            }
        }
        cv::imshow("tracked multi level", img2_single);
        cv::waitKey(1);

        chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
        // cout << "Optical Flow Pyramid costs time: " << chrono::duration_cast<chrono::duration<double>>(t2 - t1).count() << " ms" << endl;
    }

    int num_tracked = 0;
    for(const auto &s : status){
        if(s)   num_tracked++;
    }
    // cout << "number of tracked points by opticalflow: " << num_tracked <<endl;
    
    //첫번 째 frame을 key frame으로 등록 후 종료: 비교할 이미지 없음.
    if(map_->get_kfs_size() == 0){
        add_keyframe_ORB(cur_frame);        
        return 1;
    }
    else if(map_->get_kfs_size() == 1){ //아직 3D 포인트가 없으므로 2d2d pose estimation
        //새로운 키프레임 추가
        vector<DMatch> matches;
        feature_tracker_->track_feature(*last_keyframe, *cur_frame, matches);
        if(visual_odometry_->pose_estimate_2d2d(*last_keyframe, *cur_frame, matches)){
            Sophus::SE3d rel_pose = visual_odometry_->get_rel_pose();
            cur_frame->set_pose(last_keyframe->get_pose() * rel_pose);
            cout << "cur_frame pose: \n" << cur_frame->get_pose().matrix() << endl;

            visual_odometry_->triangulation(*last_keyframe, *cur_frame, matches, map_, true);
            add_keyframe_ORB(cur_frame);

            return 2;
        }
        return -1;
    }
    else{   
        VecVector3d points_3d;
        VecVector2d points_2d;
        vector<DMatch> matches;
        vector<shared_ptr<MapPoint>> inliers_mappoints;

        for(int i = 0; i<status.size(); i++){
            if(status[i]){
                success.push_back(true);
            }
            else{
                success.push_back(false);
            }
        }
        KeyPoint::convert(cur_pts2d, kp2);

        matches.clear();
        inliers_mappoints.clear();
        if(feature_tracker_->match_3d_2d_opticalflow(*prev_frame, *cur_frame, kp2, success, points_3d, points_2d, matches, inliers_mappoints) ||
            feature_tracker_->match_3d_2d(map_->mappoints(), *cur_frame, points_3d, points_2d, matches, inliers_mappoints) ||
            feature_tracker_->match_from_kf(*last_keyframe, *cur_frame, points_3d, points_2d, matches, inliers_mappoints)){
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
                const bool c1 = cur_frame->id_ >= last_keyframe->id_ + 15; 

                // 2. 트래킹 포인트가 키프레임 관측 포인트의 70% 이하이면서 15개 이상
                // PnPcompute_g2o에서 pose_inlier_mask_ 설정됨
                int num_inliers = std::count(pose_inlier_mask.begin(), pose_inlier_mask.end(), 1);
                const bool c2 = num_inliers < last_points_num * 0.7 && num_inliers > 15;
                cout << "PnP inliers: " << num_inliers << " / " << last_points_num << endl;

                // 3. 이동량 검사
                Sophus::SE3d rel = last_keyframe->get_pose().inverse() * cur_frame->get_pose();
                Sophus::SO3d R_rel = last_keyframe->get_pose().so3().inverse() * cur_frame->get_pose().so3();
                double trans = rel.translation().norm();
                double angle_rad = R_rel.log().norm();
                double angle_deg = angle_rad * 180.0 / M_PI;
                const double min_trans = 0.2;
                const double min_rot = 8.0;
                bool c_motion = (trans > min_trans) || (angle_deg > min_rot);
                cout << "Translation & Rotation since last keyframe : " << trans << " & " << angle_deg <<  endl;

                // 4. 시차 검사
                per_frame_parallax = cal_parallax_opticalflow(*prev_frame, kp2, status);
                total_parallax += per_frame_parallax;
                bool c_parrallax = total_parallax > 7.0;

                // 5. 과도한 키프레임 추가 방지
                const bool c3 = cur_frame->id_ <= last_keyframe->id_ + 5; 

                // 6. 이전 키포인트들 중 Optical flow로 추적 성공한 키포인트 비율이 많으면 키포인트 추가 X
                const bool c_tracked = (num_tracked / kp2.size()) < 0.6;

                cout << "KeyFrame conditions: " << c1 << ", " << c2 << ", " << c_motion << ", " << c_parrallax << endl;

                if((c_motion && c_parrallax)){
                    // if(num_inliers < 20 || c3 ){
                    //     cout << "Fail adding KeyFrame due to motion/parallax/inliers condition." << endl;
                    //     return 3;
                    //     // return add_keyframe(cur_frame, num_inliers);
                    // }                
                    //트래킹 포인트가 너무 적으면 키프레임 추가        
                    cout << "Adding new KeyFrame.!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << endl;
                    // 키프레임 생성과 삼각화     
                    total_parallax = 0.0;                              
                    return 0;
                }
                
                return 3;
            }
            return -1;
        }
        return -1;
    }
}


int Frontend::add_keyframe_ORB(shared_ptr<Frame> &frame){
    last_keyframe = frame;
    frame->set_keyframe(static_cast<unsigned long>(map_->get_kfs_size()));
    // TODO: 뮤텍스 필요
    map_->insert_keyframe(frame);
    cout << "key frame added" << endl;

    total_parallax = 0.0;

    return 2;
}
// 이 함수는 현재 프레임을 키프레임으로 설정하고, 삼각화 후 맵에 추가합니다.
// loftr_kps_1: 이전 키프레임의 LoFTR 키포인트들
// loftr_kps_2: 현재 프레임의 LoFTR 키포인트
// confidence: LoFTR 키포인트의 신뢰도
// mode: 키프레임 추가 모드 0: 일반 키프레임, 1: 첫 키프레임, 2: 두번째 키프레임(rescale 필요)
int Frontend::add_keyframe(const std::shared_ptr<Frame> &frame, vector<KeyPoint> &loftr_kps_1, vector<KeyPoint> &loftr_kps_2, vector<float> &confidence, int mode){
    // LoFTR 키포인트와 ORB 키포인트 매핑
    vector<pair<int, int>> loftr_orb_mappings_kf = map_loftr_to_orb(
        last_keyframe->keypoints_, loftr_kps_1);
    vector<pair<int, int>> loftr_orb_mappings_curr = map_loftr_to_orb(
        frame->keypoints_, loftr_kps_2);
    
    vector<DMatch> matches;
    // 삼각측량을 위한 DMatch 벡터 생성
    for(int i=0; i<(int)loftr_orb_mappings_kf.size(); i++){
        int orb_idx_1 = loftr_orb_mappings_kf[i].second;
        int orb_idx_2 = loftr_orb_mappings_curr[i].second;
        if(orb_idx_1 < 0 || orb_idx_2 < 0)
            continue; // 매핑이 실패한 경우 건너뜀

        DMatch m(orb_idx_1, orb_idx_2, 0.0f);
        
        // 신뢰도 기반 필터링: confidence가 0.5 이상인 경우만 사용
        if(confidence[i] >= 0.5f){
            matches.push_back(m);
        }
    }
    visual_odometry_->triangulation(*last_keyframe, *frame, matches, map_, mode==2);
    last_keyframe = frame;
    frame->set_keyframe(static_cast<unsigned long>(map_->get_kfs_size()));
    // TODO: 뮤텍스 필요
    map_->insert_keyframe(frame);
    cout << "key frame added" << endl;
    
    return 2;
}

// Optical flow로 얻은 키포인트 매칭으로 시차 계산
double Frontend::cal_parallax_opticalflow(const Frame &frame_1, const vector<KeyPoint> &kp2, const vector<uchar> &success){
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
    return visual_odometry_->check_parrallax(frame_1, *cur_frame, kp2, inliers_matches);
}

// Greedy nearest-neighbor mapping: LoFTR kp idx -> ORB kp idx (one-to-one within max_dist_px)
vector<pair<int, int>> Frontend::map_loftr_to_orb(
    const vector<KeyPoint> &orb_kps,
    const vector<KeyPoint> &loftr_kps,
    float max_dist_px) {
    vector<pair<int, int>> mappings;
    if (orb_kps.empty() || loftr_kps.empty()) {
        return mappings;
    }

    const int n_orb = static_cast<int>(orb_kps.size());
    const int k_search = std::min(5, n_orb);
    const float max_d2 = max_dist_px * max_dist_px;
    vector<bool> orb_used(n_orb, false);

    cv::Mat orb_mat(n_orb, 2, CV_32F);
    for (int i = 0; i < n_orb; ++i) {
        orb_mat.at<float>(i, 0) = orb_kps[i].pt.x;
        orb_mat.at<float>(i, 1) = orb_kps[i].pt.y;
    }
    cv::flann::Index flann_index(orb_mat, cv::flann::KDTreeIndexParams(4), cvflann::FLANN_DIST_EUCLIDEAN);

    for (int li = 0; li < static_cast<int>(loftr_kps.size()); ++li) {
        cv::Mat query(1, 2, CV_32F);
        query.at<float>(0) = loftr_kps[li].pt.x;
        query.at<float>(1) = loftr_kps[li].pt.y;

        std::vector<int> indices(k_search, -1);
        std::vector<float> dists(k_search, std::numeric_limits<float>::max());
        flann_index.knnSearch(query, indices, dists, k_search, cv::flann::SearchParams(32));

        int chosen = -1;
        float best_d2 = max_d2;
        for (int j = 0; j < k_search; ++j) {
            int oi = indices[j];
            if (oi < 0 || orb_used[oi]) continue;
            if (dists[j] <= best_d2) {
                best_d2 = dists[j];
                chosen = oi;
            }
        }
        mappings.emplace_back(li, chosen);
        if (chosen >= 0) {
            orb_used[chosen] = true;
        }
    }
    return mappings;
}