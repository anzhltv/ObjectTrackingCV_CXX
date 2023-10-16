﻿#ifndef TRALG_H_ 
#define TRALG_H_
#include "CallHistogram.h"
#include "EuclideanDistTracker.h"
#include <vector>
#include <opencv2/opencv.hpp>

//максимальное количество объектов в кадре
constexpr auto ALL_ID = 100;
//номер первого ID (-1, чтобы не совпадало ни с одним из существующих id для первого объекта)
constexpr auto FIRST_ID = -1;
//сигнал о наличии объекта во второй камере, если значение становится false - объект появился во второй камере, пока true - объекта нет  
constexpr auto REPORT = true;

class TrackingAlgorithm 
{
public:
    TrackingAlgorithm() : arrayID(ALL_ID), idSave(FIRST_ID), report(REPORT), countFrameCam(0) {};
    int idSave; // предыдущий id объекта
    bool report; // Наличие объекта в камере
    int countFrameCam; //Количество кадров проверки объекта
   
    /*
    метод для обновления трекера, получения гистограмм, поиск сравнения, выполнение алгоритма
    Input:
    detections - координаты бокса , numCam - номер камеры, frame -  кадр, countSame - количество одинаковых объектов, vectorHist - сохраненные гистограммы , tracker - трекеры объектов, trackAlg - объект класса алгоритма с другой камеры
    Output:
    бокс с верно определнным айди на кадре
    */
    void updateCameraTracking(std::vector<cv::Rect> &detections, int numCam, cv::Mat frame, int& countSame, std::vector<cv::Mat>& vectorHist, std::vector<EuclideanDistTracker>& tracker, TrackingAlgorithm& trackAlg);
    
    std::vector<int> arrayID; // Массив для накопления совпадений с конкретным объектом
private:
    int idCorrect;

    /*
    метод для определения нового объекта на полученном кадре
    Input:
    idsBoxes - координаты бокса и id нового объекта, numCam - номер камеры, frame - сам кадр, countSame - количество одинаковых объектов,
    vectorHist - сохраненные гистограммы, tracker - трекеры объектов, trackAlg - объект класса алгоритма с другой камеры
    Output:
    верно определенный id объекта и бокс на кадре
    */
    void CameraTracking(const std::vector<std::vector<int>>& idsBoxes, int numCam, const cv::Mat& frame, int& countSame,
        std::vector<cv::Mat>& vectorHist, std::vector<EuclideanDistTracker>& tracker, TrackingAlgorithm& trackAlg);

    /*
    метод для определения корректного айди для объекта
    если максимальное число совпадений больше, чем хотя бы 1/6 от количества кадров проверки,
    то отдаем объекту найденный айди,
    иначе айди по порядку
    Input:
    idGlobal - айди текущего объекта
    Output:
    корректный айди объекта
    */
    auto FindMaxSameId(int idGlobal);

    /*
    метод на случай, если найден тот же объект
    если максимальное число совпадений больше, чем хотя бы 1/6 от количества кадров проверки,
    то очищаем текущий элемент с собранным вектором для этого объекта, так как он уже есть
    + увеличиваем количество совпадающих объектов
    Input:
    arrayHist - массив с накопленными гистограммами, countSame - подсчет одинаковых элементов, idSave - айди предыдущего объекта
    Output:
    количество одинаковых элементов
    */
    int SameObjectCore(std::vector<cv::Mat>& arrayHist, int countSame, int idSave, std::vector<int>& arrID);
    int SameObject(std::vector<cv::Mat>& arrayHist, int countSame, int idSave);

    /*
    перегрузка для метода на случай, если найден тот же объект, для второй камеры, в этом случае берем массив айди из другого объекта класса
    если максимальное число совпадений больше, чем хотя бы 1/6 от количества кадров проверки,
    то очищаем текущий элемент с собранным вектором для этого объекта, так как он уже есть
    + увеличиваем количество совпадающих объектов
    Input:
    arrayHist - массив с накопленными гистограммами, countSame - подсчет одинаковых элементов, idSave - айди предыдущего объекта, arrID - массив содержащий совпадения с существующими объектами для второй камеры 
    Output:
    количество одинаковых элементов
    */
    int SameObject(std::vector<cv::Mat>& arrayHist, int countSame, int idSave, std::vector<int>& arrID);

    /*
    метод для увеличения countSame - если появился новый объект, а старый был определен к уже существующим
    Input:
    id текущего объекта по порядку, numCam, numCam2 - номер текущей и другой камеры, countSame - количество одинаковых объектов, arrayHist - сохраненные гистограммы, trackAlg - объект класса алгоритма с другой камеры
    Output:
    заполненный массив arrayID
    */
    void NewObject(int id, int numCam, int numCam2, int& countSame, std::vector<cv::Mat>& arrayHist, TrackingAlgorithm& trackAlg);
};

#endif //TRALG