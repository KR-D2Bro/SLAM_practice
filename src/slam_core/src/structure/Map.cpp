#include "slam_core/Map.hpp"

#include "slam_core/frame.hpp"
#include "slam_core/mappoint.hpp"

using namespace std;

void Map::insert_keyframe(shared_ptr<Frame> keyframe){
    lock_guard<mutex> lk(kf_mutex_);
    keyframes_.push_back(keyframe);
    kf_id2ptr_.emplace(keyframe->keyframe_id_, keyframe);
}

void Map::insert_mappoint(shared_ptr<MapPoint> mappoint){
    std::lock_guard<std::mutex> lk(mp_mutex_);
    mappoints_.push_back(mappoint);
    mp_id2ptr_.emplace(mappoint->id_, mappoint);
}

bool Map::modify_keyframe_pose(unsigned long id, Sophus::SE3d new_pose){
    lock_guard<mutex> lk(kf_mutex_);
    if(kf_id2ptr_.find(id) != kf_id2ptr_.end()){
        kf_id2ptr_[id]->set_pose(new_pose);
        return true;
    }
    return false;
}

bool Map::modify_mappoint_pos(unsigned long id, Vec3 new_pos){
    lock_guard<mutex> lk(mp_mutex_);
    if(mp_id2ptr_.find(id) != mp_id2ptr_.end()){
        mp_id2ptr_[id]->set_pos(new_pos);
        return true;
    }
    return false;
}

// 맵포인트 그룹 가져오는 함수
vector<shared_ptr<MapPoint>> Map::get_mpGroup(const vector<shared_ptr<Frame>> &keyframe_group) {
    vector<shared_ptr<MapPoint>> mappoint_group;
    vector<int> mp_counts(get_mps_size(), 0); // 중복 제거를 위한 집합
    int threshold = static_cast<int>(keyframe_group.size()*0.5);  // 최소 관측 키프레임 수

    // 각 키프레임에서 관측된 맵포인트들 카운팅
    for(const auto& kf : keyframe_group){
        for(const auto& mp : kf->observed_map_points_){
            if(mp){
                mp_counts[mp->id_]++;;
            }
        }
    }

    for(int i = 0; i< mp_counts.size(); i++){
        if(mp_counts[i] > threshold){
            auto mp = mappoints_[i];
            if(mp){
                mappoint_group.push_back(mp);
            }
        }
    }

    return mappoint_group;
}

// 키프레임 그룹 가져오는 함수
// TODO:현재 벡터의 인덱스로 접근 -> 추후 id 기반 접근으로 변경 필요
vector<shared_ptr<Frame>> Map::get_kfGroup(const shared_ptr<Frame> &kf, int radius) {
    vector<shared_ptr<Frame>> keyframe_group;
    keyframe_group.push_back(kf);
    for(int i = 1; i <= radius; i++){
        auto prev_kf = get_keyframe_at(static_cast<int>(kf->keyframe_id_) - i);
        
        if(prev_kf) {
            keyframe_group.push_back(prev_kf);
        }
        else {
            break;
        }
    }
    return keyframe_group;
}
