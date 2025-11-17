#include "slam_core/opticalflow_tracker.hpp"
#include "slam_core/frame.hpp"
#include <algorithm>

using namespace std;
using namespace cv;

void OpticalFlowSingleLevel(
    const Mat &img1,
    const Mat &img2,
    const vector<KeyPoint> &kp1,
    vector<KeyPoint> &kp2,
    vector<bool> &success,
    bool inverse, bool has_initial);

// Implement pyramidical Lucas-Kanade Optical Flow Tracker
void OpticalFlowTracker::track_opticalflow(
    const Frame &frame1,
    const Frame &frame2,
    vector<KeyPoint> &kp2,
    vector<DMatch> &matches,
    std::vector<bool> &success,
    bool inverse,
    bool has_initial) 
{
    img1_= frame1.img_, img2_ = frame2.img_; 
    success_ = &success;
    inverse_ = inverse, has_initial_ = has_initial;

    sort(matches.begin(), matches.end(), [](const DMatch &m1, const DMatch &m2){
        return m1.distance < m2.distance;
    });
    if(matches.size() > 10) matches.resize(10); //신뢰도 상위 10개 매칭만 사용

    success_->resize(matches.size(), true);

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

    Mat img2_single;
    cv::cvtColor(img2_, img2_single, cv::COLOR_GRAY2BGR);
    for (int i = 0; i < kp2_.size(); i++) {
        if (success_->at(i)) {
            cv::circle(img2_single, kp2_[i].pt, 2, cv::Scalar(0, 250, 0), 2);
            cv::line(img2_single, kp1_[i].pt, kp2_[i].pt, cv::Scalar(0, 250, 0));
        }
    }
    cv::imshow("tracked single level", img2_single);
    cv::waitKey(1);
}

void OpticalFlowTracker::track_pyramid_opticalflow(
    Frame &frame1,
    Frame &frame2,
    vector<KeyPoint> &kp2,
    vector<bool> &success,
    bool inverse) 
{
    img1_= frame1.img_, img2_ = frame2.img_; 
    success_ = &success;
    inverse_ = inverse;
    kp1_ = frame1.keypoints_;

    // parameters
    int pyramids = 4;
    double pyramid_scale = 0.5;
    double scales[] = {1.0, 0.5, 0.25, 0.125};

    // create pyramids
    vector<Mat> pyr1, pyr2; // image pyramids
    for (int i = 0; i < pyramids; i++) {
        if (i == 0) {
            pyr1.push_back(img1_);
            pyr2.push_back(img2_);
        } else {
            Mat img1_pyr, img2_pyr;
            cv::resize(pyr1[i - 1], img1_pyr,
                       cv::Size(pyr1[i - 1].cols * pyramid_scale, pyr1[i - 1].rows * pyramid_scale));
            cv::resize(pyr2[i - 1], img2_pyr,
                       cv::Size(pyr2[i - 1].cols * pyramid_scale, pyr2[i - 1].rows * pyramid_scale));
            pyr1.push_back(img1_pyr);
            pyr2.push_back(img2_pyr);
        }
    }

    // coarse-to-fine LK tracking in pyramids
    vector<KeyPoint> kp1_pyr, kp2_pyr;
    for (auto &kp:kp1_) {
        auto kp_top = kp;
        kp_top.pt *= scales[pyramids - 1];
        kp1_pyr.push_back(kp_top);
        kp2_pyr.push_back(kp_top);
    }

    for (int level = pyramids - 1; level >= 0; level--) {
        // from coarse to fine
        success.clear();
        OpticalFlowSingleLevel(pyr1[level], pyr2[level], kp1_pyr, kp2_pyr, success, inverse, true);

        if (level > 0) {
            for (auto &kp: kp1_pyr)
                kp.pt /= pyramid_scale;
            for (auto &kp: kp2_pyr)
                kp.pt /= pyramid_scale;
        }
    }

    for(int i = 0; i<kp2_pyr.size(); i++){
        if(success[i] == false)
            continue;
        kp2.push_back(kp2_pyr[i]);
    }

    Mat img2_single;
    cv::cvtColor(img2_, img2_single, cv::COLOR_GRAY2BGR);
    for (int i = 0; i < kp2.size(); i++) {
        if (success_->at(i)) {
            cv::circle(img2_single, kp2[i].pt, 2, cv::Scalar(0, 250, 0), 2);
            cv::line(img2_single, kp1_[i].pt, kp2[i].pt, cv::Scalar(0, 250, 0));
        }
    }
    cv::imshow("tracked multi level", img2_single);
    cv::waitKey(1);
} 

bool OpticalFlowTracker::calculateOpticalFlow(
    const Range &range
){
    int half_patch_size = 4;
    int iterations = 10;
    for(size_t i = range.start; i < range.end; i++){
        auto kp = kp1_[i];
        double dx=0, dy=0;
        if(has_initial_){
            dx = kp2_[i].pt.x - kp.pt.x;
            dy = kp2_[i].pt.y - kp.pt.y;
        }

        double cost = 0, lastCost = 0;
        bool succ = true;

        // Gauss-Newton iterations
        Eigen::Matrix2d H = Eigen::Matrix2d::Zero();
        Eigen::Vector2d b = Eigen::Vector2d::Zero();
        Eigen::Vector2d J[2*half_patch_size][2*half_patch_size];  // Jacobian을 patch 내 모든 픽셀에 대한 vector로 정의해야됨
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
                    double error = GetPixelValue(img1_, kp.pt.x + x, kp.pt.y + y) - 
                                   GetPixelValue(img2_, kp.pt.x + x + dx, kp.pt.y + y + dy);
                    
                    if(inverse_ == false){
                        // inverse 모드가 아닐 때에는 매 iteration마다 라이브 이미지에서 도함수를 계산
                        J[half_patch_size + x][half_patch_size + y] = -1.0 * Eigen::Vector2d(
                            0.5 * (GetPixelValue(img2_, kp.pt.x + x + dx + 1, kp.pt.y + dy + y) - GetPixelValue(img2_, kp.pt.x + x + dx - 1, kp.pt.y + dy + y)),
                            0.5 * (GetPixelValue(img2_, kp.pt.x + x + dx, kp.pt.y + y + dy + 1) - GetPixelValue(img2_, kp.pt.x + x + dx, kp.pt.y + y + dy - 1))
                        );
                    } else if(iter == 0){
                        //inverse 모드에서는 J가 모든 iteration에서 동일
                        J[half_patch_size + x][half_patch_size + y] = -1.0 * Eigen::Vector2d(
                            0.5 * (GetPixelValue(img1_, kp.pt.x + x + 1, kp.pt.y + y) - GetPixelValue(img1_, kp.pt.x + x - 1, kp.pt.y + y)),
                            0.5 * (GetPixelValue(img1_, kp.pt.x + x, kp.pt.y + y + 1) - GetPixelValue(img1_, kp.pt.x + x, kp.pt.y + y - 1))
                        );
                    }
                    
                    b += -error * J[half_patch_size + x][half_patch_size + y];
                    cost += error * error;
                    if(inverse_ == false || iter == 0){
                        H += J[half_patch_size + x][half_patch_size + y] * J[half_patch_size + x][half_patch_size + y].transpose();
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

        if(!succ){
            continue;
        }
        kp2_[i].pt = kp.pt + Point2f(dx, dy);
    }
}

void OpticalFlowSingleLevel(
    const Mat &img1,
    const Mat &img2,
    const vector<KeyPoint> &kp1,
    vector<KeyPoint> &kp2,
    vector<bool> &success,
    bool inverse, bool has_initial) {
    kp2.resize(kp1.size());
    success.resize(kp1.size());
    OpticalFlowTracker opticalFlowTracker(img1, img2, kp1, kp2, success, inverse, has_initial);
    parallel_for_(Range(0, kp1.size()),
                  std::bind(&OpticalFlowTracker::calculateOpticalFlow, &opticalFlowTracker, placeholders::_1));
                  
    kp2 = opticalFlowTracker.tracked_keypoints();
    success = opticalFlowTracker.get_success();
}
