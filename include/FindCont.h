#ifndef OBJTR_H_ 
#define OBJTR_H_
#include <opencv2/opencv.hpp>

class FindCont 
{
public:
    FindCont();
    std::vector<cv::Rect> GettingCoordinates(cv::Mat frame1, cv::Mat frame2, int kern, int iter, int size);
private:
    void DetectContour(int size, std::vector<std::vector<cv::Point>> contours, std::vector<cv::Rect>& detections);
    auto Subtractor(const cv::Mat frame1, const cv::Mat frame2, int kern, int iter);
};

#endif //OBJTR_H_