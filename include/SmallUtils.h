#ifndef UTILS_H_ 
#define UTILS_H_

#include <iostream>
#include <opencv2/opencv.hpp>



// Метод для вычисления площади прямоугольника
// Вход:
// x1, y1, x2, y2 - координаты прямоугольника
// Выход:
// Площадь прямоугольника
int calculate_area(int x1, int y1, int x2, int y2) 
{
    return (x2 - x1) * (y2 - y1);
}

// Метод для коррекции отображения ID
// Если прямоугольник касается верхней границы, текст отображается под прямоугольником, иначе над ним
// Вход:
// y, h - координата левого верхнего угла и высота прямоугольника
// Выход:
// Скорректированная координата y
int if_border(int y, int h) 
{
    return (y < 50) ? (y + h + 30) : (y - 15);
}


// Метод для нахождения центральной точки
// Вход:
// Координаты прямоугольника
// Выход:
// Координаты центра
std::pair<int, int> center_point_save(int x1, int x2, int y1, int y2) 
{
    return std::make_pair((x1 + x2) / 2, (y1 + y2) / 2);
}


// Метод для изменения размера и объединения кадров в один кадр
// Вход:
// Кадры с камер и размер экрана
// Выход:
// Объединенный кадр
void resize_frame(cv::Mat frame1, cv::Mat frame2, int width, int height) 
{
    cv::Mat Combi;
    cv::hconcat(frame1, frame2, Combi);
    cv::resize(Combi, Combi, cv::Size(width * 4 / 3, height * 2 / 3));
    cv::imshow("Combined", Combi);
}

#endif //UTILS_H_