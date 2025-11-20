#ifndef MAP_HPP_
#define MAP_HPP_

#include<slam_core/common_include.hpp>

class Frame;
class MapPoint;

class Map{
    public:
        Map() {}

        void insert_keyframe(std::shared_ptr<Frame> keyframe);

        void insert_mappoint(std::shared_ptr<MapPoint> mappoint);

        const std::vector<std::shared_ptr<Frame>>& keyframes() const{
            return keyframes_;
        }

        const std::vector<std::shared_ptr<MapPoint>>& mappoints() const{
            return mappoints_;
        }

        std::shared_ptr<Frame> get_keyframe_at(int index) const {
            if(index < 0 || index >= keyframes_.size()) {
                return nullptr;
            }
            return keyframes_[index];
        }

        const int get_kfs_size() const {
            return static_cast<int>(keyframes_.size());
        }

        const int get_mps_size() const {
            return static_cast<int>(mappoints_.size());
        }

        bool modify_keyframe_pose(unsigned long id, Sophus::SE3d new_pose);
        bool modify_mappoint_pos(unsigned long id, Vec3 new_pos);

        // 맵포인트 그룹 가져오는 함수
        std::vector<std::shared_ptr<MapPoint>> get_mpGroup(const std::vector<std::shared_ptr<Frame>> &keyframe_group);
        // 키프레임 그룹 가져오는 함수
        std::vector<std::shared_ptr<Frame>> get_kfGroup(const std::shared_ptr<Frame> &kf, int radius);

    private:
        std::vector<std::shared_ptr<Frame>> keyframes_;
        std::vector<std::shared_ptr<MapPoint>> mappoints_;
        std::mutex kf_mutex_, mp_mutex_;
        std::unordered_map<unsigned long, std::shared_ptr<Frame>> kf_id2ptr_;
        std::unordered_map<unsigned long, std::shared_ptr<MapPoint>> mp_id2ptr_;
};

#endif