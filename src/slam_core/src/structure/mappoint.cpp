#include "slam_core/mappoint.hpp"
#include "slam_core/frame.hpp"

using namespace std;

void MapPoint::remove_observation(std::shared_ptr<Frame> frame){
    observations_.remove_if([frame](const std::weak_ptr<Frame>& wp) {
        auto sp = wp.lock();
        return sp && sp.get() == frame.get();
    });
    observed_cnt_ = static_cast<int>(observations_.size());
}