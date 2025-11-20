#include "slam_core/mappoint.hpp"
#include "slam_core/frame.hpp"

using namespace std;

void MapPoint::remove_observation(std::shared_ptr<Frame> frame){
    auto iter = observations_.find(frame.get());
    if(iter != observations_.end()){
        observations_.erase(iter);
        observed_cnt_ = static_cast<int>(observations_.size());
    }
}
