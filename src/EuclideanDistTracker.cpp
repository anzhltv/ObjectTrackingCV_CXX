#include "EuclideanDistTracker.h"
#include <iostream>
#include <cmath>

constexpr int MAX_DIST = 100;
constexpr int MAX_DIST_ = 150;

EuclideanDistTracker::EuclideanDistTracker() : idCount(0) {}

/*
метод для трекинга на одном кадре
Input
objectsRect - x,y,w,h объекта 
Output
координаты объекта и верно определенный id объекта
*/
std::vector<std::vector<int>> EuclideanDistTracker::update(const std::vector<cv::Rect>& objectsRect) 
{
    std::vector<std::vector<int>> objectsBbsIds;

    for (auto rect : objectsRect) 
    {
        int x = rect.x;
        int y = rect.y;
        int w = rect.width;
        int h = rect.height;
        //центры нового объекта 
        int cx = (x + x + w) / 2;
        int cy = (y + y + h) / 2;

        auto sameObjectDetected = false;
        auto dist = 0.0;

        for (auto& kv : centerPoints) 
        {
            int id = kv.first;
            cv::Point2i pt = kv.second;
            //подсчет евклидового расстояния 
            dist = std::sqrt(std::pow(cx - pt.x, 2) + std::pow(cy - pt.y, 2));
            //если расстояние меньше определенной величины, то добавляем координаты к предыдущему id и sameObjectDetected=True
            if (dist < MAX_DIST) {
                centerPoints[id] = cv::Point2i(cx, cy);
                std::cout << "\n{" << id << ": (" << centerPoints[id].x << ", " << centerPoints[id].y << ")}";
                objectsBbsIds.push_back({ x, y, w, h, id });
                sameObjectDetected = true;
                break;
            }
        }
        //если sameObjectDetected=False, то заносим под новым id и увеличиваем счетчик 
        if (!sameObjectDetected) 
        {
            centerPoints[idCount] = cv::Point2i(cx, cy);
            objectsBbsIds.push_back({ x, y, w, h, idCount });
            idCount++;
        }
    }

    std::map<int, cv::Point2i> newCenterPoints;
    //сохраняем центр текущего объекта 
    for (auto & objBbId : objectsBbsIds) 
    {
        auto objectId = objBbId[4];
        cv::Point2i center = centerPoints[objectId];
        newCenterPoints[objectId] = center;
    }

    centerPoints = newCenterPoints;
    return objectsBbsIds;
}