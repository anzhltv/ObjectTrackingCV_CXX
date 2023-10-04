#include "FindCont.h"

//параметры (размер ядра) для прямоугольного структурирующего элемента getStructuringElement и дальнейшего dilate
constexpr int KERNEl[] = { 2, 3 };
//параметры (количество итераций) для операции расширения 
constexpr int ITER[] = { 3, 7 };
//значение порога, используемое для сравнения с пиксельными значениями на входном изображении
constexpr auto thresholdValue = 20;
//значение, которым будет заменено значение пикселей, проходящих порог 
constexpr auto maxVal = 255;

FindCont::FindCont() {}

/*
 метод для получение маски 
 Input:
 frame1 - кадр с которым производится вычитание, frame2 - новый кадр , kern - размер ядра, iter - количество итераций
 Output:
 заполненный вектор detections
*/
auto FindCont::Subtractor(const cv::Mat &frame1, const cv::Mat &frame2, int num_cam) 
{
    cv::Mat diff;
    cv::absdiff(frame1, frame2, diff);
    cv::cvtColor(diff, diff, cv::COLOR_BGR2GRAY);
    cv::Mat mask;
    cv::threshold(diff, mask, thresholdValue, maxVal, cv::THRESH_BINARY);
    auto kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(KERNEl[num_cam], KERNEl[num_cam]));
    cv::dilate(mask, mask, kernel, cv::Point(-1, -1), ITER[num_cam]);
    return mask;
}


/*
 метод для получение кординат бокса с объектом
 Input:
 size - минимальный размер бокса, contours - контур объекта, detections - вектор для записи результата
 Output:
 заполненный вектор detections
*/
void FindCont::DetectContour(int size, const std::vector<std::vector<cv::Point>> &contours, std::vector<cv::Rect>& detections) 
{
    for (const auto& cnt : contours) {
        double area = cv::contourArea(cnt);
        if (area > size) {
            auto bounding_rect = cv::boundingRect(cnt);
            detections.push_back(bounding_rect);
        }
    }
}


/*
 метод для для объединения метода Substractor и DetectContour
 Input:
 frame1 - кадр с которым производится вычитание, frame2 - новый кадр , kern - размер ядра, iter - количество итераций, size - минимальный размер бокса
 Output:
 заполненный вектор detections
*/
std::vector<cv::Rect> FindCont::GettingCoordinates(const cv::Mat &frame1, const cv::Mat &frame2, int num_cam, int size)
{
    cv::Mat mask;
    mask = Subtractor(frame1, frame2, num_cam);
    std::vector<std::vector<cv::Point>> contours;
    findContours(mask, contours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);
    std::vector<cv::Rect> detections;
    DetectContour(size, contours, detections);
    return detections;
}