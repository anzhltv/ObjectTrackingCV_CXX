#pragma once

#include <vector>
#include <map>
#include <opencv2/core.hpp>

class EuclideanDistTracker 
{
public:
    int idCount;
    EuclideanDistTracker();
    std::vector<std::vector<int>> update(std::vector<cv::Rect>& objectsRect);
private:
    std::map<int, cv::Point2i> centerPoints;
};