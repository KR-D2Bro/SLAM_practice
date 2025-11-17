#ifndef OPTICALFLOW_TRACKER_HPP_
#define OPTICALFLOW_TRACKER_HPP_

#include<slam_core/common_include.hpp>

class Frame;

class OpticalFlowTracker{
    public:
    OpticalFlowTracker(){}

    OpticalFlowTracker(const cv::Mat &img1, const cv::Mat &img2, const std::vector<cv::KeyPoint> &kp1, std::vector<cv::KeyPoint> &kp2,
        std::vector<bool> &success, bool inverse = true, bool has_initial = false)
        : img1_(img1), img2_(img2), kp1_(kp1), kp2_(kp2), success_(&success),
        inverse_(inverse), has_initial_(has_initial) {}

    ~OpticalFlowTracker(){
    }

    const std::vector<cv::KeyPoint>& tracked_keypoints() const { return kp2_; }
    const std::vector<bool>& get_success() const { return *success_; }

    void track_opticalflow(
        const Frame &frame1,
        const Frame &frame2,
        std::vector<cv::KeyPoint> &kp2,
        std::vector<cv::DMatch> &matches,
        std::vector<bool> &success,
        bool inverse = true,
        bool has_initial = false
    );

    bool calculateOpticalFlow(const cv::Range &range);

    void track_pyramid_opticalflow(
    Frame &frame1,
    Frame &frame2,
    std::vector<cv::KeyPoint> &kp2,
    std::vector<bool> &success,
    bool inverse);

    private:

    private:
    cv::Mat img1_, img2_;
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
