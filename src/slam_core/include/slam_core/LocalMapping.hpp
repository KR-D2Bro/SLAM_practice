#ifndef LOCAL_MAPPING_HPP_
#define LOCAL_MAPPING_HPP_

#include<slam_core/common_include.hpp>
#include <condition_variable>

class Map;
class Frame;
class MapPoint;

class LocalMapping{
    public:
        LocalMapping(std::shared_ptr<Map> &map, cv::Mat &K) : map_(map), running_(true) {
            cv::cv2eigen(K, K_);
            worker_ = std::thread(&LocalMapping::Process, this);
        }

        ~LocalMapping(){
            {
                std::lock_guard<std::mutex> lk(queue_mutex_);
                running_ = false;
            }  
            cv_.notify_all();
            worker_.join();
        }

        void insert_kf2queue(std::shared_ptr<Frame> &keyframe){
            {
                std::lock_guard<std::mutex> lk(queue_mutex_);
                keyframe_queue_.push(keyframe);
            }
            cv_.notify_one();
        }

    private:
        void Process();
        
        bool local_bundle_adjustment(const std::vector<std::shared_ptr<Frame>> &keyframe_group, const std::vector<std::shared_ptr<MapPoint>> &mappoint_group);
        
        std::shared_ptr<Map> map_;
        std::thread worker_;
        std::mutex queue_mutex_;
        std::condition_variable cv_;
        std::queue<std::shared_ptr<Frame>> keyframe_queue_;
        bool running_;
        Eigen::Matrix3d K_;
};


#endif