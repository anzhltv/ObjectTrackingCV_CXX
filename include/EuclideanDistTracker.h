#pragma once

#include <vector>
#include <map>
#include <opencv2/core.hpp>

class EuclideanDistTracker 
{
public:
    int idCount; //подсчет объектов в камере
    EuclideanDistTracker();
    /*
    метод для трекинга на одном кадре
    Input
    objectsRect - x,y,w,h объекта
    Output
    координаты объекта и верно определенный id объекта
    */
    std::vector<std::vector<int>> update(const std::vector<cv::Rect>& objectsRect); //обновление трекера 
private:
    // map для сохранения предыдущего центра объекта
    std::map<int, cv::Point2i> centerPoints;
};