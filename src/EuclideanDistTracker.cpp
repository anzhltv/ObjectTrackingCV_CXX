#include "EuclideanDistTracker.h"
#include <iostream>
#include <cmath>

constexpr int MAX_DIST = 100;
constexpr int MAX_DIST_ = 150;

EuclideanDistTracker::EuclideanDistTracker() : idCount(0) {}

std::vector<std::vector<int>> EuclideanDistTracker::update(std::vector<cv::Rect>& objectsRect) 
{
    std::vector<std::vector<int>> objectsBbsIds;

    for (cv::Rect rect : objectsRect) 
    {
        int x = rect.x;
        int y = rect.y;
        int w = rect.width;
        int h = rect.height;
        int cx = (x + x + w) / 2;
        int cy = (y + y + h) / 2;

        bool sameObjectDetected = false;
        double dist = 0.0;

        for (auto& kv : centerPoints) 
        {
            int id = kv.first;
            cv::Point2i pt = kv.second;
            dist = std::sqrt(std::pow(cx - pt.x, 2) + std::pow(cy - pt.y, 2));

            if (dist < MAX_DIST) {
                centerPoints[id] = cv::Point2i(cx, cy);
                std::cout << "\n{" << id << ": (" << centerPoints[id].x << ", " << centerPoints[id].y << ")}";
                objectsBbsIds.push_back({ x, y, w, h, id });
                sameObjectDetected = true;
                break;
            }
        }

        if (!sameObjectDetected) 
        {
            centerPoints[idCount] = cv::Point2i(cx, cy);
            objectsBbsIds.push_back({ x, y, w, h, idCount });
            idCount++;
        }
    }

    std::map<int, cv::Point2i> newCenterPoints;

    for (std::vector<int>& objBbId : objectsBbsIds) 
    {
        int objectId = objBbId[4];
        cv::Point2i center = centerPoints[objectId];
        newCenterPoints[objectId] = center;
    }

    centerPoints = newCenterPoints;
    return objectsBbsIds;
}