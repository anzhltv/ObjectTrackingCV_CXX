#include "FindCont.h"



FindCont::FindCont() {}

/*Subtractor
 ����� ��� ��������� �������� ����� � ��������
 Input:
 frame1 - ���� � ������� ������������ ���������, frame2 - ����� ���� , kern - ������ ����, iter - ���������� ��������
 Output:
 ����������� ������ detections
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
 ����� ��� ��������� �������� ����� � ��������
 Input:
 size - ����������� ������ �����, contours - ������ �������, detections - ������ ��� ������ ����������
 Output:
 ����������� ������ detections
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
 ����� ��� ��� ����������� ������ Substractor � DetectContour
 Input:
 frame1 - ���� � ������� ������������ ���������, frame2 - ����� ���� , kern - ������ ����, iter - ���������� ��������, size - ����������� ������ �����
 Output:
 ����������� ������ detections
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