#ifndef CALLH_H_ 
#define CALLH_H_
#include <opencv2/opencv.hpp>
#include <vector>

void CallHistogram(cv::Mat frameHistNew, std::vector<cv::Mat>& arrHist, int idGlobal, double optParam, std::vector<int>& arrID);

#endif //CALLH_H_