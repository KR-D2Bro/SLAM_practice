#include "slam_core/opticalflow_tracker.hpp"
#include "slam_core/frame.hpp"

using namespace std;
using namespace cv;

// Implement pyramidical Lucas-Kanade Optical Flow Tracker
void OpticalFlowTracker::track_opticalflow(
    Frame &frame1,
    Frame &frame2,
    vector<KeyPoint> &kp2,
    vector<DMatch> &matches,
    std::vector<bool> &success,
    bool inverse,
    bool has_initial) 
{
    img1_= &frame1.img_, img2_ = &frame2.img_; 
    success_ = &success;
    inverse_ = inverse, has_initial_ = has_initial;

    success_->assign(matches.size(), true);

    kp1_.resize(matches.size());
    kp2_.resize(matches.size());
    for(int i = 0; i< matches.size(); i++){
        kp1_[i] = frame1.keypoints_[matches[i].queryIdx];
        kp2_[i] = frame2.keypoints_[matches[i].trainIdx];
    }

    cv::parallel_for_(Range(0, matches.size() ),
        std::bind(&OpticalFlowTracker::calculateOpticalFlow, this, std::placeholders::_1));

    vector<bool> isUpdated(kp2_.size(), false);
    for(int i = 0; i<kp2_.size(); i++){
        if(success_->at(i) == false || isUpdated[i])
            continue;
        kp2[matches[i].trainIdx] = kp2_[i];
        isUpdated[i] = true;
    }

    Mat img2_single = img2_->clone();
    for (int i = 0; i < kp2_.size(); i++) {
        if (success_->at(i)) {
            cv::circle(img2_single, kp2_[i].pt, 2, cv::Scalar(0, 250, 0), 2);
            cv::line(img2_single, kp1_[i].pt, kp2_[i].pt, cv::Scalar(0, 250, 0));
        }
    }
    cv::imshow("tracked single level", img2_single);
    cv::waitKey(1);
} 

bool OpticalFlowTracker::calculateOpticalFlow(
    const Range &range
){
    int half_patch_size = 4;
    int iterations = 10;
    for(size_t i = range.start; i < range.end; i++){
        auto kp = kp1_[i];
        double dx, dy;
        dx = kp2_[i].pt.x - kp.pt.x;
        dy = kp2_[i].pt.y - kp.pt.y;

        double cost = 0, lastCost = 0;
        bool succ = true;

        // Gauss-Newton iterations
        Eigen::Matrix2d H = Eigen::Matrix2d::Zero();
        Eigen::Vector2d b = Eigen::Vector2d::Zero();
        Eigen::Vector2d J;
        for(int iter = 0; iter < iterations; iter++){
            //inverse : LK optical flow의 inverse Compositional 방식인지 아닌지를 결정하는 것으로 매 반복마다 gradient를 다시 계산할지 말지를 결정.
            if(inverse_ == false){   
                H = Eigen::Matrix2d::Zero();
                b = Eigen::Vector2d::Zero();
            } else{
                b = Eigen::Vector2d::Zero();
            }
            
            cost = 0;

            for(int x = -half_patch_size; x < half_patch_size; x++){
                for(int y = -half_patch_size; y < half_patch_size; y++){
                    double error = GetPixelValue(*img1_, kp.pt.x + x, kp.pt.y + y) - 
                                   GetPixelValue(*img2_, kp.pt.x + x + dx, kp.pt.y + y + dy);
                    
                    if(inverse_ == false){
                        // inverse 모드가 아닐 때에는 매 iteration마다 라이브 이미지에서 도함수를 계산
                        J = -1.0 * Eigen::Vector2d(
                            0.5 * (GetPixelValue(*img2_, kp.pt.x + x + dx + 1, kp.pt.y + dy + y) - GetPixelValue(*img2_, kp.pt.x + x + dx - 1, kp.pt.y + dy + y)),
                            0.5 * (GetPixelValue(*img2_, kp.pt.x + x + dx, kp.pt.y + y + dy + 1) - GetPixelValue(*img2_, kp.pt.x + x + dx, kp.pt.y + y + dy - 1))
                        );
                    } else if(iter == 0){
                        //inverse 모드에서는 J가 모든 iteration에서 동일
                        J = -1.0 * Eigen::Vector2d(
                            0.5 * (GetPixelValue(*img1_, kp.pt.x + x + 1, kp.pt.y + y) - GetPixelValue(*img1_, kp.pt.x + x - 1, kp.pt.y + y)),
                            0.5 * (GetPixelValue(*img1_, kp.pt.x + x, kp.pt.y + y + 1) - GetPixelValue(*img1_, kp.pt.x + x, kp.pt.y + y - 1))
                        );
                    }
                    
                    b += -error * J;
                    cost += error * error;
                    if(inverse_ == false || iter == 0){
                        H += J * J.transpose();
                    }
                }
            }

            Eigen::Vector2d update = H.ldlt().solve(b);

            if(std::isnan(update[0])){
                cout << "update is nan" << endl;
                succ = false;
                break;
            }

            if(iter > 0 && cost > lastCost){
                break;
            }

            // update dx, dy
            dx += update[0];
            dy += update[1];
            lastCost = cost;
            succ = true;

            // 수렴한 경우
            if(update.norm() < 1e-2){
                break;
            }
        }

        success_->at(i) = succ;

        kp2_[i].pt = kp.pt + Point2f(dx, dy);
        
    }
}