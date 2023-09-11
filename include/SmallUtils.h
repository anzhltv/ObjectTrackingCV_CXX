#ifndef UTILS_H_ 
#define UTILS_H_

#include <iostream>
#include <opencv2/opencv.hpp>



// ����� ��� ���������� ������� ��������������
// ����:
// x1, y1, x2, y2 - ���������� ��������������
// �����:
// ������� ��������������
int calculate_area(int x1, int y1, int x2, int y2) 
{
    return (x2 - x1) * (y2 - y1);
}

// ����� ��� ��������� ����������� ID
// ���� ������������� �������� ������� �������, ����� ������������ ��� ���������������, ����� ��� ���
// ����:
// y, h - ���������� ������ �������� ���� � ������ ��������������
// �����:
// ����������������� ���������� y
int if_border(int y, int h) 
{
    return (y < 50) ? (y + h + 30) : (y - 15);
}


// ����� ��� ���������� ����������� �����
// ����:
// ���������� ��������������
// �����:
// ���������� ������
std::pair<int, int> center_point_save(int x1, int x2, int y1, int y2) 
{
    return std::make_pair((x1 + x2) / 2, (y1 + y2) / 2);
}


// ����� ��� ��������� ������� � ����������� ������ � ���� ����
// ����:
// ����� � ����� � ������ ������
// �����:
// ������������ ����
void resize_frame(cv::Mat frame1, cv::Mat frame2, int width, int height) 
{
    cv::Mat Combi;
    cv::hconcat(frame1, frame2, Combi);
    cv::resize(Combi, Combi, cv::Size(width * 4 / 3, height * 2 / 3));
    cv::imshow("Combined", Combi);
}

#endif //UTILS_H_