#include "CallHistogram.h"

constexpr auto EPSILON = 1e-6;
constexpr auto ALPHA = 0.1;
const int HIST_SIZE[] = { 160, 160, 160 };
float range[] = { 0, 256 };
const float* HIST_RANGE[] = { range, range, range };
const int CHANNELS[] = { 0, 1, 2 };

namespace
{

cv::Mat EqualizeHistFrame(cv::Mat frameHistNew)
{
    cv::Mat frameHist;
    std::vector<cv::Mat> channelsEq;
    cv::split(frameHistNew, channelsEq);
    cv::equalizeHist(channelsEq[0], channelsEq[0]);
    cv::equalizeHist(channelsEq[1], channelsEq[1]);
    cv::equalizeHist(channelsEq[2], channelsEq[2]);
    cv::merge(channelsEq, frameHist);
    return frameHist;
}

/*image_hist
 метод для накопления гистограмм
 Input:
 frameHistNew - бокс с объектом, arrHist - массив куда накапливаются гистограммы объектов, idGlobal - id текущего объекта по порядку
 Output:
 заполненный массив arrHist
*/
void ImageHist(cv::Mat frameHistNew, std::vector<cv::Mat>& arrHist, int idGlobal) 
{
    auto frameHist = EqualizeHistFrame(frameHistNew);

    cv::Mat hist;
    cv::calcHist(&frameHist, 1, CHANNELS, cv::Mat(), hist, 3, HIST_SIZE, HIST_RANGE, true, false);

    cv::normalize(hist, hist, 0, 1.0, cv::NORM_MINMAX);

    if (cv::norm(arrHist[idGlobal], cv::NORM_L2) < EPSILON) 
    {
        arrHist[idGlobal] = hist.clone();
    }
    else 
    {
        arrHist[idGlobal] = ALPHA * hist + (1 - ALPHA) * arrHist[idGlobal];
    }
}


/*SearchCompare
 метод подсчета совпадений
 Input:
 idGlobal - id текущего объекта по порядку, optParam - наименьший параметр схожести,
 arrID - массив, содержащий количество совпадений с каждым существующим id
 Output:
 заполненный массив arrID
*/
void SearchCompare(int idGlobal, double optParam, std::vector<int>& arrID, std::vector<cv::Mat>& arrayHist) 
{
    int objID = 0, objK = 0;
    double max = 0;
    double compareHistCor;
    
    for (int i = 0; i < idGlobal; ++i) 
    {
        try 
        {
            compareHistCor = cv::compareHist(arrayHist[idGlobal], arrayHist[i], cv::HISTCMP_CORREL);
        }
        catch (cv::Exception& e) 
        {
            std::cout << e.what() << std::endl;
            compareHistCor = 0;
        }
        if (compareHistCor > optParam && compareHistCor != 1) 
        {
            if (max < compareHistCor) 
            {
                max = compareHistCor;
                objID = i;
                ++objK;
            }
        }
    }
    if (objK != 0) 
    {
        ++arrID[objID];
    }
}

}


/*
 Метод для накопления гистограмм и поиска сравнения новой гистограммы с уже имеющимимся
 Input:
 frameHistNew - бокс с объектом, arrHist - массив куда накапливаются гистограммы объектов, idGlobal - id текущего объекта по порядку, 
 optParam - наименьший параметр схожести, arrID - массив, содержащий количество совпадений с каждым существующим id
 Output:
 заполненный массив arrHist и arrID 
*/
void CallHistogram(cv::Mat frameHistNew, std::vector<cv::Mat>& arrHist, int idGlobal, double optParam, std::vector<int>& arrID)
{
    ImageHist(frameHistNew, arrHist, idGlobal);
    SearchCompare(idGlobal, optParam, arrID, arrHist);
}