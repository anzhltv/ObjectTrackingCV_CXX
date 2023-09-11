#ifndef TRALG_H_ 
#define TRALG_H_
#include "CallHistogram.h"
#include "EuclideanDistTracker.h"
#include <vector>
#include <opencv2/opencv.hpp>

class TrackingAlgorithm 
{
public:
    TrackingAlgorithm() : arrayID(100), idSave(-1), report(true), countFrameCam(0) {};
    int idSave; // ���������� id �������
    bool report; // ������� ������� � ������
    int countFrameCam; //���������� ������ �������� �������
    void updateCameraTracking(std::vector<cv::Rect> detections, int numCam, cv::Mat frame, int& countSame, std::vector<cv::Mat>& vectorHist, std::vector<EuclideanDistTracker>& tracker, TrackingAlgorithm& trackAlg);
    std::vector<int> arrayID; // ������ ��� ���������� ���������� � ���������� ��������
private:
    int idCorrect;
    void CameraTracking(std::vector<std::vector<int>> idsBoxes, int numCam, cv::Mat frame, int& countSame,
        std::vector<cv::Mat>& vectorHist, std::vector<EuclideanDistTracker>& tracker, TrackingAlgorithm& trackAlg);
    auto FindMaxSameId(int idGlobal);
    int SameObject(std::vector<cv::Mat>& arrayHist, int countSame, int idSave);
    int SameObject(std::vector<cv::Mat>& arrayHist, int countSame, int idSave, std::vector<int>& arrID);
    void NewObject(int id, int numCam, int numCam2, int& countSame, std::vector<cv::Mat>& arrayHist, TrackingAlgorithm& trackAlg);
};

#endif //TRALG