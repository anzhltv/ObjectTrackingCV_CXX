#include "FindCont.h"



FindCont::FindCont() {}

/*Subtractor
 метод для получение кординат бокса с объектом
 Input:
 frame1 - кадр с которым производится вычитание, frame2 - новый кадр , kern - размер ядра, iter - количество итераций
 Output:
 заполненный вектор detections
*/
auto FindCont::Subtractor(const cv::Mat frame1, const cv::Mat frame2, int kern, int iter) 
{
    cv::Mat diff;
    cv::absdiff(frame1, frame2, diff);
    cv::cvtColor(diff, diff, cv::COLOR_BGR2GRAY);
    cv::Mat mask;
    cv::threshold(diff, mask, 20, 255, cv::THRESH_BINARY);
    auto kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(kern, kern));
    cv::dilate(mask, mask, kernel, cv::Point(-1, -1), iter);
    return mask;
}


/*DetectContour
 метод для получение кординат бокса с объектом
 Input:
 size - минимальный размер бокса, contours - контур объекта, detections - вектор для записи результата
 Output:
 заполненный вектор detections
*/
void FindCont::DetectContour(int size, std::vector<std::vector<cv::Point>> contours, std::vector<cv::Rect>& detections) 
{
    for (const auto& cnt : contours) {
        double area = cv::contourArea(cnt);
        if (area > size) {
            auto bounding_rect = cv::boundingRect(cnt);
            detections.push_back(bounding_rect);
        }
    }
}


/*GettingCoordinates
 метод для для объединения метода Substractor и DetectContour
 Input:
 frame1 - кадр с которым производится вычитание, frame2 - новый кадр , kern - размер ядра, iter - количество итераций, size - минимальный размер бокса
 Output:
 заполненный вектор detections
*/
std::vector<cv::Rect> FindCont::GettingCoordinates(cv::Mat frame1, cv::Mat frame2, int kern, int iter, int size)
{
    cv::Mat mask;
    mask = Subtractor(frame1, frame2, kern, iter);
    std::vector<std::vector<cv::Point>> contours;
    findContours(mask, contours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);
    std::vector<cv::Rect> detections;
    DetectContour(size, contours, detections);
    return detections;
}