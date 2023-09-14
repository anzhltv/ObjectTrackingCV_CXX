#ifndef OBJTR_H_ 
#define OBJTR_H_
#include <opencv2/opencv.hpp>

class FindCont 
{
public:
    FindCont();

    /*
    метод для для объединения метода Substractor и DetectContour
    Input:
    frame1 - кадр с которым производится вычитание, frame2 - новый кадр , kern - размер ядра, iter - количество итераций, size - минимальный размер бокса
    Output:
    заполненный вектор detections
    */
    std::vector<cv::Rect> GettingCoordinates(const cv::Mat& frame1, const cv::Mat& frame2, int num_cam, int size);
private:
    /*
    метод для получение кординат бокса с объектом
    Input:
    size - минимальный размер бокса, contours - контур объекта, detections - вектор для записи результата
    Output:
    заполненный вектор detections
    */
    void DetectContour(int size, const std::vector<std::vector<cv::Point>> &contours, std::vector<cv::Rect>& detections);

    /*
    метод для получение маски
    Input:
    frame1 - кадр с которым производится вычитание, frame2 - новый кадр , kern - размер ядра, iter - количество итераций
    Output:
    заполненный вектор detections
    */
    auto Subtractor(const cv::Mat &frame1, const cv::Mat &frame2, int num_cam);
};

#endif //OBJTR_H_