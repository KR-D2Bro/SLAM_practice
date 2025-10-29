#include "slam_core/visual_odometry.hpp"
#include<iostream>
#include<memory>

using namespace std;
using namespace cv;

VisualOdometry::VisualOdometry(cv::Mat &K){
    this->K = K;
}

Sophus::SE3d VisualOdometry::get_rel_pose(){
    return T_.inverse();
}

bool VisualOdometry::pose_estimate_2d2d(const Frame &frame_1, const Frame &frame_2, const std::vector<cv::DMatch> &matches){
    vector<Point2f> points1, n_points1;
    vector<Point2f> points2, n_points2;

    pose_inlier_mask_.clear();
    const size_t kMinMatches = 8;
    if (matches.size() < kMinMatches) {
        cout << "Too few matches for 2d-2d pose estimation: " << matches.size() << endl;
        return false;
    }

    for(int i=0; i < (int)matches.size(); i++){
        points1.push_back(frame_1.keypoints_[matches[i].queryIdx].pt);
        points2.push_back(frame_2.keypoints_[matches[i].trainIdx].pt);
    }

    Mat fundamental_matrix, homography_matrix_21;
    fundamental_matrix = findFundamentalMat(points1, points2, FM_RANSAC, 1.0);
    homography_matrix_21 = findHomography(points1, points2, cv::RANSAC, 1.0);

    float score_F = calc_fundamental_score(points1, points2, fundamental_matrix);
    float score_H = calc_homography_score(points1, points2, homography_matrix_21); 
    cout << "score_F : " << score_F << endl << "score_H : " << score_H << endl;

    if(score_H / (score_H + score_F) > 0.45){
        return false;
    }
    
    Mat essential_matrix;
    essential_matrix = findEssentialMat(points1, points2, K);
    // cv::Mat essential_matrix_hand = K.t() * fundamental_matrix * K;
    cout << "E: "<< essential_matrix <<endl;
    // cout << "E_hand: "<< essential_matrix_hand <<endl;

    Mat inlier_mask;
    int inliers = recoverPose(essential_matrix, points1, points2, K, R, t, inlier_mask);

    if(inliers < 50){
        cout << "inliers is too small: " << inliers << endl;
        return false;
    }
    cout << "inliers is enough: " << inliers << endl;

    pose_inlier_mask_.clear();
    pose_inlier_mask_.reserve(inlier_mask.rows * inlier_mask.cols);
    for(int r = 0; r < inlier_mask.rows; ++r){
        for(int c = 0; c < inlier_mask.cols; ++c){
            pose_inlier_mask_.push_back(inlier_mask.at<uchar>(r, c));
        }
    }

    cout << "R: " << R << endl << "t: " << t << endl;

    //calculate T_
    //T_는 cam1 좌표계의 포인트들을 cam2좌표계로 변환하는 행렬
    //T1 -> T2 로의 상대 Pose는 T_.inverse()이다.
    Eigen::Matrix<double,3,3> rot;
    Eigen::Matrix<double,3,1> tran;
    cv2eigen(R,rot);
    cv2eigen(t,tran);
    T_ = Sophus::SE3d(rot, tran);

    cout << "T_rel: " << T_.matrix3x4() << endl;

    return true;
}

float VisualOdometry::calc_homography_score(const vector<Point2f> &points1, const vector<Point2f> &points2, cv::Mat &H_21){
    float score = 0.0;
    const float chi_sq_threshold_H = 5.991; //homography의 카이제곱 임계값 : 자유도 2
    cv::Mat H_12 = H_21.inv();

    for(int i = 0; i < (int)points1.size(); i++){
        //1번 frame의 point의 1번 img 평면상의 픽셀 좌표 homogeneous
        cv::Mat point1_h = (cv::Mat_<double>(3,1) << points1[i].x, points1[i].y, 1);
        //1번 frame의 point를 2번 img 평면으로 projection한 픽셀 좌표
        cv::Mat point1_proj_h = H_21 * point1_h;
        point1_proj_h /= point1_proj_h.at<double>(2);
        cv::Point2f point1_proj(
            static_cast<float>(point1_proj_h.at<double>(0)),
            static_cast<float>(point1_proj_h.at<double>(1))
        );
        float error = cv::norm(points2[i] - point1_proj) * cv::norm(points2[i] - point1_proj);
        if(error < chi_sq_threshold_H){
            score += chi_sq_threshold_H - error;
        }

        //2번 프레임의 point를 1번으로 proj해서 오차 계산
        cv::Mat point2_h = (cv::Mat_<double>(3,1) << points2[i].x, points2[i].y, 1);
        //2번 frame의 point를 2번 img 평면으로 projection한 픽셀 좌표
        cv::Mat point2_proj_h = H_12 * point2_h;
        point2_proj_h /= point2_proj_h.at<double>(2);
        cv::Point2f point2_proj(point2_proj_h.at<double>(0), point2_proj_h.at<double>(1));
        error = cv::norm(points1[i] - point2_proj) * cv::norm(points1[i] - point2_proj);
        if(error < chi_sq_threshold_H){
            score += chi_sq_threshold_H - error;
        }
    }

    return score;
}

float VisualOdometry::calc_fundamental_score(const vector<Point2f> &points1, const vector<Point2f> &points2, cv::Mat &F_21){
    float score = 0.0;
    const float chi_sq_threshold_F = 3.841; //F의 카이제곱 임계값 : 자유도 1
    cv::Mat F_21_t = F_21.t();

    cout <<F_21.type() <<endl;

    for(int i=0; i< (int)points1.size(); i++){
        cv::Mat tmp = F_21 * (cv::Mat_<double>(3,1) << (double)points1[i].x, (double)points1[i].y, 1);
        
        cv::Vec3f line1_proj = cv::Vec3f(tmp);
        float error = std::abs(points2[i].x * line1_proj[0] + points2[i].y * line1_proj[1] + line1_proj[2]) / sqrt(line1_proj[0]*line1_proj[0] + line1_proj[1]*line1_proj[1]);
        if(error*error < chi_sq_threshold_F){
            score += chi_sq_threshold_F - error*error;
        }

        tmp = F_21_t * (cv::Mat_<double>(3,1) << points2[i].x, points2[i].y, 1);
        cv::Vec3f line2_proj = cv::Vec3f(tmp);
        error = std::abs(points1[i].x * line2_proj[0] + points1[i].y * line2_proj[1] + line2_proj[2]) / sqrt(line2_proj[0]*line2_proj[0] + line2_proj[1]*line2_proj[1]);
        if(error*error < chi_sq_threshold_F){
            score += chi_sq_threshold_F - error*error;
        }
    }

    return score;
}

//상대 좌표를 이용한 방법
bool VisualOdometry::triangulation(const Frame &frame_1, Frame &frame_2, 
                                    const std::vector<cv::DMatch> &matches, 
                                    std::vector<shared_ptr<MapPoint>> &points, bool isFirst
                                    ){
    // --- 1. 기존 삼각측량 준비 ---
    cv::Mat T1 = (cv::Mat_<double>(3, 4) <<
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0);

    T1 = K * T1; //projection matrix
    
    cv::Mat T21;
    Sophus::SE3d T21_se3 = frame_2.get_pose().inverse() * frame_1.get_pose();
    // eigen2cv(T_.matrix3x4(), T21);
    eigen2cv(T21_se3.matrix3x4(), T21);
    T21 = K * T21;

    const bool has_pose_mask = !pose_inlier_mask_.empty();

    std::vector<int> inlier_match_indices;
    std::vector<cv::Point2f> pts_1, pts_2;
    for(int i=0;i<matches.size();i++){
        if(has_pose_mask && pose_inlier_mask_[i] == 0)
            continue;

        pts_1.push_back(frame_1.keypoints_[matches[i].queryIdx].pt);
        pts_2.push_back(frame_2.keypoints_[matches[i].trainIdx].pt);
        inlier_match_indices.push_back(static_cast<int>(i));
    }

    cv::Mat pts_4d;
    cv::triangulatePoints(T1, T21, pts_1, pts_2, pts_4d);

    std::vector<cv::Point3d> valid_points_local; // frame_1 좌표계 기준 유효 포인트
    std::vector<int> valid_match_indices;

    for (int i = 0; i < pts_4d.cols; i++) {
        cv::Mat x = pts_4d.col(i);
        x /= x.at<float>(3, 0); // 4D -> 3D
        cv::Point3d p(static_cast<double>(x.at<float>(0, 0)), static_cast<double>(x.at<float>(1, 0)), static_cast<double>(x.at<float>(2, 0)));

        // 검증 1: 첫 번째 카메라 앞에서 포인트가 유효한가 (depth > 0)
        if (p.z > 0) {
            // 검증 2: 두 번째 카메라 좌표계로 변환했을 때도 유효한가
            Eigen::Vector3d p_eigen(p.x, p.y, p.z);
            Eigen::Vector3d p_in_frame2 = T21_se3 * p_eigen;
            if (p_in_frame2.z() > 0) {
                // 재투영 오차가 크다면 폐기
                Vec2 proj1 = cam2pixel(p_eigen, K);
                Vec2 proj2 = cam2pixel(p_in_frame2, K);
                cv::Point2f obs1 = pts_1[i];
                cv::Point2f obs2 = pts_2[i];

                double err1 = cv::norm(cv::Point2f(proj1.x(), proj1.y()) - obs1);
                double err2 = cv::norm(cv::Point2f(proj2.x(), proj2.y()) - obs2);
                const double MAX_REPROJ_ERR = 3.0;
                if(err1 > MAX_REPROJ_ERR || err2 > MAX_REPROJ_ERR){
                    continue;
                }

                // 모든 검증 통과, 포인트 추가
                valid_points_local.push_back(p);
                valid_match_indices.push_back(inlier_match_indices[i]);
            }
        }
    }
    cout << "Valid points after cheirality check: " << valid_points_local.size() << endl;
    
    if(isFirst){
        rescaleInitialMap(valid_points_local, T21_se3);
        frame_2.set_pose(frame_1.get_pose() * T21_se3.inverse());
    }

    // 3. 최종 월드 좌표계 포인트 계산
    vector<Vec3> world_points;
    vector<shared_ptr<MapPoint>> new_map_points;
    int start_id = points.size();
    for(size_t k = 0; k < valid_points_local.size(); ++k){
        const auto& p_local = valid_points_local[k];
        const auto& match = matches[valid_match_indices[k]];
        Eigen::Vector3d p_eigen_local(p_local.x, p_local.y, p_local.z);
        Eigen::Vector3d p_world = frame_1.get_pose() * p_eigen_local;

        shared_ptr<MapPoint> mp = MapPoint::CreateNewMappoint(start_id + k, 
                    p_world,
                    frame_1.descriptors_.row(match.queryIdx).clone());
        
        new_map_points.push_back(mp);

        // 각 프레임에 관측된 맵포인트 추가
        frame_2.observed_map_points_.push_back(mp);
        // 맵 포인트에 관측 프레임 추가, 이후에 기존에 있는 mappoint와의 중복 관측 제거 필요
        try {
            auto frame2_ptr = frame_2.shared_from_this();
            mp->observations_.push_back(frame2_ptr);
        } catch (const std::bad_weak_ptr &) {
        }

        try {
            auto frame1_ptr_const = frame_1.shared_from_this();
            mp->observations_.push_back(std::const_pointer_cast<Frame>(frame1_ptr_const));
        } catch (const std::bad_weak_ptr &) {
        }
        mp->observed_cnt_ = static_cast<int>(mp->observations_.size());
    }

    frame_2.observed_map_points_ = new_map_points;
    points.insert(points.end(), new_map_points.begin(), new_map_points.end());

    return true; // 성공
}

//triangulation으로 계산한 포인트들의 중앙 depth로 스케일 정규화
bool VisualOdometry::rescaleInitialMap(std::vector<Point3d> &world_points, Sophus::SE3d &T_21){
    vector<double> depths;
    double median_depth = 0.0;
    for(auto &p : world_points){
        depths.push_back(p.z);
    }

    if(depths.empty()){
        return false; // 유효 포인트 없음
    }
    sort(depths.begin(), depths.end());
    median_depth = depths[depths.size() / 2];
    cout << "Median depth: " << median_depth << endl;
    
    if(median_depth <= 1e-6){
        return false; // depth가 너무 작음
    }

    for(auto &pt : world_points){
        pt /= median_depth; 
    }
    T_21.translation() /= median_depth;

    return true;
}

//G2o 라이브러리를 이용한 PnP 최적화
bool VisualOdometry::PnPcompute_g2o(const VecVector3d &points_3d, const VecVector2d &points_2d, Frame &cur_frame){
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, Eigen::Dynamic>> BlockSolverType;   // pose is 6, landmark is 3
    typedef g2o::LinearSolverEigen<BlockSolverType::PoseMatrixType> LinearSolverType;   // Gradient descent type
    
    Eigen::Matrix3d K_eigen;
    cv2eigen(K, K_eigen);
    auto solver = new g2o::OptimizationAlgorithmGaussNewton(
        std::make_unique<BlockSolverType>(std::make_unique<LinearSolverType>())
    );
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(true);

    VertexPose *vertex_pose = new VertexPose();
    vertex_pose->setId(0);
    vertex_pose->setEstimate(cur_frame.get_pose().inverse());
    optimizer.addVertex(vertex_pose);

    int index = 0;
    for(size_t i = 0; i < points_2d.size();i++){
        auto p2d = points_2d[i];
        auto p3d = points_3d[i];
        EdgeProjection *edge = new EdgeProjection(p3d, K_eigen);
        edge->setId(index);
        edge->setVertex(0, vertex_pose);
        edge->setMeasurement(p2d);
        edge->setInformation(Eigen::Matrix2d::Identity());

        auto huber = new g2o::RobustKernelHuber();
        huber->setDelta(1.0);
        edge->setRobustKernel(huber);

        optimizer.addEdge(edge);
        index++;
    }

    optimizer.setVerbose(true);
    optimizer.initializeOptimization();
    optimizer.optimize(10);

    cur_frame.set_pose(vertex_pose->estimate().inverse());

    // 최종 inlier 마스크 계산
    // 3D 포인트와 2D 포인트의 재투영 오차 기반으로 outlier 라는 것은 오매칭일 가능성이 높다는 것을 의미
    pose_inlier_mask_.clear();
    pose_inlier_mask_.resize(points_2d.size(), 1);
    const double chi2_threshold = 5.99; // (≈ 2.45px)^2 or 원하는 값

    for (auto edge_ptr : optimizer.edges()) {
        auto *edge = dynamic_cast<EdgeProjection*>(edge_ptr);
        double chi2 = edge->chi2();
        if (chi2 > chi2_threshold) {
            pose_inlier_mask_[edge->id()] = 0;
        }
    }

    return true;
}

bool VisualOdometry::check_parrallax(const Frame &frame_1, const Frame &frame_2, const std::vector<cv::DMatch> &matches, double min_parallax_deg){
    Eigen::Matrix3d K_eigen;
    cv::cv2eigen(K, K_eigen);
    Eigen::Matrix3d K_inv = K_eigen.inverse();

    std::vector<double> angles_deg;
    angles_deg.reserve(matches.size());
    for(const auto& m : matches){
        cv::Point2f p1_cv = frame_1.keypoints_[m.queryIdx].pt;
        cv::Point2f p2_cv = frame_2.keypoints_[m.trainIdx].pt;

        Eigen::Vector3d p1_h(p1_cv.x, p1_cv.y, 1.0);
        Eigen::Vector3d p2_h(p2_cv.x, p2_cv.y, 1.0);

        Eigen::Vector3d f1 = K_inv * p1_h;
        f1.normalize();
        Eigen::Vector3d f2 = K_inv * p2_h;
        f2.normalize();

        double cos_angle = f1.dot(f2);
        cos_angle = std::max(-1.0, std::min(1.0, cos_angle));
        double angle_rad = std::acos(cos_angle);
        angles_deg.push_back(angle_rad * 180.0 / M_PI);
    }

    const size_t mid = angles_deg.size() / 2;
    std::nth_element(angles_deg.begin(), angles_deg.begin() + mid, angles_deg.end());
    double median_angle = angles_deg[mid];
    
    cout << "Median parallax angle : " << median_angle << " degrees" << endl;
    return median_angle >= min_parallax_deg;
}
