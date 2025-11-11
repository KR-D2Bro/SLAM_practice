#ifndef OPTICALFLOW_TRACKER_HPP_
#define OPTICALFLOW_TRACKER_HPP_

#include<slam_core/common_include.hpp>

class Frame;

class OpticalFlowTracker{
    public:
    OpticalFlowTracker(){}

    ~OpticalFlowTracker(){
    }

    void track_opticalflow(
        Frame &frame1,
        Frame &frame2,
        std::vector<cv::KeyPoint> &kp2,
        std::vector<cv::DMatch> &matches,
        std::vector<bool> &success,
        bool inverse = true,
        bool has_initial = false
    );

    private:
    bool calculateOpticalFlow(const cv::Range &range);

    private:
    cv::Mat *img1_, *img2_;
    std::vector<cv::KeyPoint> kp1_, kp2_;
    std::vector<bool> *success_;
    bool inverse_ = true;
    bool has_initial_ = false;

    // 선형 보간법을 사용하여 이미지의 (x, y) 위치에서 픽셀 값을 얻습니다.
    inline float GetPixelValue(const cv::Mat &img, float x, float y) {
        // boundary check
        if (x < 0) x = 0;
        if (y < 0) y = 0;
        if (x >= img.cols - 1) x = img.cols - 2;
        if (y >= img.rows - 1) y = img.rows - 2;
        
        float xx = x - floor(x);
        float yy = y - floor(y);
        int x_a1 = std::min(img.cols - 1, int(x) + 1);
        int y_a1 = std::min(img.rows - 1, int(y) + 1);
        
        return (1 - xx) * (1 - yy) * img.at<uchar>(y, x)
        + xx * (1 - yy) * img.at<uchar>(y, x_a1)
        + (1 - xx) * yy * img.at<uchar>(y_a1, x)
        + xx * yy * img.at<uchar>(y_a1, x_a1);
    }
};



#endif