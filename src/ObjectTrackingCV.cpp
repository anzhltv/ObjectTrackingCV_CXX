#include "FindCont.h"
#include "TrackingAlgorithm.h"
#include <opencv2/opencv.hpp>
#include <chrono>
#include <vector>
#include <iostream>

// параметры минимальных размеров боксов
constexpr auto PERSENT_SIZE_BOX_L = 0.045;
constexpr auto PERSENT_SIZE_BOX_R = 0.0076;
// параметры для обрезки кадра со второй камеры
constexpr auto PARAM_ROI_B = 4.2; 
constexpr auto PARAM_ROI_E = 2.5; 
// номера камер
constexpr int NUM_CAM[] = { 0, 1 };
// параметры для отображения видео
constexpr auto PARAM_ROI_W = 1.3;
constexpr auto PARAM_ROI_H = 0.7;


int main() 
{
    std::vector<EuclideanDistTracker> tracker(2); // Создание двух объектов класса EuclideanDistTracker для трекинга
    std::vector<TrackingAlgorithm> trackAlg(2); // Создание двух объектов класса TrackingAlgorithm для выполнения алгоритма

    std::string path1 = "C:/video/Camera3.avi";
    std::string path2 = "C:/video/Camera4.avi";

    cv::VideoCapture cap1(path1);
    cv::VideoCapture cap2(path2);
    cv::Mat frame1_1, frame2_2; // Кадры для первой и второй камеры
    cap1.read(frame1_1);
    cap2.read(frame2_2);

    const int width = static_cast<int>(cap1.get(cv::CAP_PROP_FRAME_WIDTH));
    const int height = static_cast<int>(cap1.get(cv::CAP_PROP_FRAME_HEIGHT));

    frame2_2 = frame2_2(cv::Rect(int(width / PARAM_ROI_B), 0, int(width / PARAM_ROI_E), height)); // Обрезка кадра

    const auto size1 = static_cast<int>(PERSENT_SIZE_BOX_L * width * height); // Минимальные размеры боксов
    const auto size2 = static_cast<int>(PERSENT_SIZE_BOX_R * width * height);

    std::vector<cv::Mat> vector_hist(100);
    auto count_same = 0; // Переменная для подсчета одинаковых объектов

    if (!cap1.isOpened())
    {
        std::cout << "Error opening video stream or file" << std::endl;
    }
    else
    {
        const auto fps = static_cast<int>(cap1.get(cv::CAP_PROP_FPS));
        std::cout << "Frames per second: " << fps;
    }

    while (cap1.isOpened())
    {
        setlocale(LC_ALL, "Russian");

        cv::Mat frame1, mask1, mask2;
        cv::Mat frame2, roi2;

        bool isSuccess1 = cap1.read(frame1);
        bool isSuccess2 = cap2.read(frame2);

        roi2 = frame2(cv::Rect(int(width / PARAM_ROI_B), 0, int(width / PARAM_ROI_E), height));

        FindCont findCont;

        /*GettingCoordinates
        метод для для объединения метода Substractor и DetectContour
        Input:
        frame1 - кадр с которым производится вычитание, frame2 - новый кадр , kern - размер ядра, iter - количество итераций, size - минимальный размер бокса
        Output:
        заполненный вектор detections
        */
        auto detections1 = findCont.GettingCoordinates(frame1_1, frame1, NUM_CAM[0], size1);
        auto detections2 = findCont.GettingCoordinates(frame2_2, roi2, NUM_CAM[1], size2);

        /*
        метод для обновления трекера, получения гистограмм, поиск сравнения, выполнение алгоритма
        Input:
        detections - координаты бокса , numCam - номер камеры, frame -  кадр, countSame - количество одинаковых объектов, vectorHist - сохраненные гистограммы , tracker - трекеры объектов, trackAlg - объект класса алгоритма с другой камеры
        Output:
        бокс с верно определнным айди на кадре
        */
        trackAlg[0].updateCameraTracking(detections1, NUM_CAM[0], frame1, count_same, vector_hist, tracker, trackAlg[1]);
        trackAlg[1].updateCameraTracking(detections2, NUM_CAM[1], roi2, count_same, vector_hist, tracker, trackAlg[0]);

        if (isSuccess1 && isSuccess2) 
        {
            cv::Mat Combi;
            cv::hconcat(frame1, frame2, Combi);
            cv::resize(Combi, Combi, cv::Size(width * PARAM_ROI_W, height * PARAM_ROI_H));
            cv::imshow("Combined", Combi);
        }

        if (!isSuccess1 && !isSuccess2) 
        {
            std::cout << "End of video" << std::endl;
            break;
        }

        int key = cv::waitKey(1);
        if (key == 'q') 
        {
            std::cout << "q key is pressed by the user. Stopping the video" << std::endl;
            break;
        }
    }
    return 0;
}