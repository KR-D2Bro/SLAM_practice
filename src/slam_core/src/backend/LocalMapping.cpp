#include "slam_core/LocalMapping.hpp"

#include "slam_core/frame.hpp"
#include "slam_core/G2oTypes.hpp"
#include "slam_core/Map.hpp"
#include "slam_core/mappoint.hpp"

using namespace std;

void LocalMapping::Process(){
    while(true){
        cout << "LocalMapping: waiting for new keyframe..." << endl;
        shared_ptr<Frame> kf;
        {
            std::unique_lock<std::mutex> lk(queue_mutex_);
            cv_.wait(lk, [&]{ return !keyframe_queue_.empty() || !running_; });
            if(!running_ && keyframe_queue_.empty()) return;
            kf = keyframe_queue_.front();
            keyframe_queue_.pop();
            cout << "LocalMapping: new keyframe popped from queue. id: " << kf->keyframe_id_ << endl;
        }

        // 이전 5개 키프레임을 가져온다.
        auto keyframe_group = map_->get_kfGroup(kf, 5);

        // 키프레임 그룹에 해당하는 맵포인트들을 가져온다.
        auto mappoint_group = map_->get_mpGroup(keyframe_group);

        // TODO: kf를 이용한 local mapping 작업 수행
        local_bundle_adjustment(keyframe_group, mappoint_group);

        cout << "LocalMapping: local bundle adjustment done for keyframe id: " << kf->keyframe_id_ << endl;
    }
}




bool LocalMapping::local_bundle_adjustment(const vector<shared_ptr<Frame>> &keyframe_group, const vector<shared_ptr<MapPoint>> &mappoint_group) {
    // 키프레임이 2개 미만이면 수행하지 않음
    if(keyframe_group.size() < 3)   return false; 

    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 3>> BlockSolverType;
    typedef g2o::LinearSolverCSparse<BlockSolverType::PoseMatrixType> LinearSolverType;

    auto solver = new g2o::OptimizationAlgorithmLevenberg(
        std::make_unique<BlockSolverType>(std::make_unique<LinearSolverType>())
    );
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(true);

    vector<VertexPose *> vertex_poses;
    vector<VertexPoint *> vertex_points;
    // 키프레임들 vertex 추가
    for(int i = 0; i < keyframe_group.size(); i++){
        VertexPose *vertex_pose = new VertexPose();
        vertex_pose->setId(i);
        if(i == keyframe_group.size()-1){
            vertex_pose->setFixed(true); // 가장 예전 키프레임은 고정
        }
        vertex_pose->setEstimate(keyframe_group[i]->get_pose().inverse());
        optimizer.addVertex(vertex_pose);
        vertex_poses.push_back(vertex_pose);
    }
    // 맵포인트들 vertex 추가
    for(int i=0; i< mappoint_group.size(); i++){
        VertexPoint *vertex_point = new VertexPoint();
        vertex_point->setId(i + keyframe_group.size()); // 키프레임 id와 겹치지 않도록
        vertex_point->setEstimate(mappoint_group[i]->pos_);
        vertex_point->setMarginalized(true);
        optimizer.addVertex(vertex_point);
        vertex_points.push_back(vertex_point);
    }
    // 엣지 추가
    int edge_id = 0;
    for(int i=0; i < mappoint_group.size(); i++){
        auto mp = mappoint_group[i];
        for(int kf_idx=0; kf_idx<keyframe_group.size(); kf_idx++){
            auto &frame = keyframe_group[kf_idx];
            auto iter = mp->observations_.find(frame.get());
            if(iter != mp->observations_.end()){
                int kp_idx = iter->second;
                auto &kp = frame->keypoints_[kp_idx];
                EdgeProjectionBA *edge = new EdgeProjectionBA(K_);
                edge->setId(edge_id++);
                edge->setVertex(0, vertex_poses[kf_idx]);
                edge->setVertex(1, vertex_points[i]);
                edge->setMeasurement(Vec2(kp.pt.x, kp.pt.y));
                edge->setInformation(Eigen::Matrix2d::Identity());
                edge->setRobustKernel(new g2o::RobustKernelHuber());
                optimizer.addEdge(edge);
            }
        }
    }

    optimizer.initializeOptimization();
    optimizer.optimize(10);

    // TODO: 최적화된 결과를 다시 맵포인트와 키프레임에 반영
    // map_ 클래스 내부 함수로 업데이트
    for(int i=0; i< keyframe_group.size(); i++){
        auto &frame = keyframe_group[i];
        auto vertex_pose = vertex_poses[i];
        Sophus::SE3d optimized_pose = vertex_pose->estimate().inverse();
        
        map_->modify_keyframe_pose(frame->keyframe_id_, optimized_pose);
    }
    for(int i=0; i< mappoint_group.size(); i++){
        auto &mp = mappoint_group[i];
        auto vertex_point = vertex_points[i];
        Vec3 optimized_pos = vertex_point->estimate();
        
        map_->modify_mappoint_pos(mp->id_, optimized_pos);
    }

    return true;
}