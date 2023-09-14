#ifndef CALLH_H_ 
#define CALLH_H_
#include <opencv2/opencv.hpp>
#include <vector>

/*
 Метод для накопления гистограмм и поиска сравнения новой гистограммы с уже имеющимимся
 Input:
 frameHistNew - бокс с объектом, arrHist - массив куда накапливаются гистограммы объектов, idGlobal - id текущего объекта по порядку,
 optParam - наименьший параметр схожести, arrID - массив, содержащий количество совпадений с каждым существующим id
 Output:
 заполненный массив arrHist и arrID
*/
void CallHistogram(const cv::Mat& frameHistNew, std::vector<cv::Mat>& arrHist, int idGlobal, double optParam, std::vector<int>& arrID);

#endif //CALLH_H_