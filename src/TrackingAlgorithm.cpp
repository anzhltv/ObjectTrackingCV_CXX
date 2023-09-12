#include "TrackingAlgorithm.h"
#include "SmallUtils.h"


constexpr auto COUNT_FRAME = 40; // Количество кадров для идентификации человека
const std::vector<float> OPT_PARAM = { 0.5f, 0.21f }; // Границы сравнения векторов для каждого кадра
constexpr auto PART_FRAME = 6; // Часть от общего числа кадров проверки


/* FindMaxSameId
 метод для определения корректного айди для объекта
 если максимальное число совпадений больше, чем хотя бы 1/6 от количества кадров проверки,
 то отдаем объекту найденный айди,
 иначе айди по порядку
 Input:
 idGlobal - айди текущего объекта
 Output:
 корректный айди объекта
*/
auto TrackingAlgorithm::FindMaxSameId(int idGlobal) 
{
    std::vector<int> arrayID_list(arrayID.begin(), arrayID.end());
    // Если максимальное совпадение больше чем 1/6 от общего числа кадров проверки
    if (*std::max_element(arrayID_list.begin(), arrayID_list.end()) >= COUNT_FRAME / PART_FRAME)
    {
        // То берем найденный id
        const auto idCorr = static_cast<int>(std::distance(arrayID_list.begin(), std::max_element(arrayID_list.begin(), arrayID_list.end())));
        return idCorr;
    }
    else 
    {
        // Иначе следующий по порядку
        const auto idCorr = idGlobal;
        return idCorr;
    }
}



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
int TrackingAlgorithm::SameObject(std::vector<cv::Mat>& arrayHist, int countSame, int idSave) 
{
    //если был найден существующий объект, то очистка собранного вектора и массива id + увеличение числа совпадающих объектов
    auto maxEl = arrayID[0];
    for (auto i = 1; i < arrayID.size(); ++i) 
    {
        if (arrayID[i] > maxEl) 
        {
            maxEl = arrayID[i];
        }
    }
    if (maxEl >= COUNT_FRAME / PART_FRAME)
    {
        try
        {
            arrayHist[idSave - countSame] = cv::Mat();
        }
        catch (const std::exception& e)
        {
            std::cout << "\n Error! " << e.what();
        }
        
        ++countSame;
    }
    std::for_each(arrayID.begin(), arrayID.end(), [](int& value) {
        value = 0;
        });
    //arrayID.clear();
    return countSame;
}

/*
 перегрузка для метода на случай, если найден тот же объект, для второй камеры, в этом случае берем массив айди из другого объекта класса
 если максимальное число совпадений больше, чем хотя бы 1/6 от количества кадров проверки,
 то очищаем текущий элемент с собранным вектором для этого объекта, так как он уже есть
 + увеличиваем количество совпадающих объектов
 Input:
arrayHist - массив с накопленными гистограммами, countSame - подсчет одинаковых элементов, idSave - айди предыдущего объекта
 Output:
 количество одинаковых элементов
 */
int TrackingAlgorithm::SameObject(std::vector<cv::Mat>& arrayHist, int countSame, int idSave, std::vector<int>& arrID)
{
    //если был найден существующий объект, то очистка собранного вектора и массива id + увеличение числа совпадающих объектов
    auto maxEl = arrID[0];
    for (auto i = 1; i < arrID.size(); ++i)
    {
        if (arrID[i] > maxEl) 
        {
            maxEl = arrID[i];
        }
    }
    if (maxEl >= COUNT_FRAME / PART_FRAME)
    {
        try
        {
            arrayHist[idSave - countSame] = cv::Mat();
        }
        catch (const std::exception& e)
        {
            std::cout << "\n Error! " << e.what();
        }

        ++countSame;
    }
    std::for_each(arrID.begin(), arrID.end(), [](int& value) {
        value = 0;
        });
    //arrID.clear();
    return countSame;
}


/*
 метод для увеличения countSame - если появился новый объект, а старый был определен к уже существующим
 Input:
 id текущего объекта по порядку, numCam, numCam2 - номер текущей и другой камеры, countSame - количество одинаковых объектов, arrayHist - сохраненные гистограммы, trackAlg - объект класса алгоритма с другой камеры
 Output:
 заполненный массив arrayID
*/
void TrackingAlgorithm::NewObject(int id, int numCam, int numCam2, int& countSame, std::vector<cv::Mat>& arrayHist, TrackingAlgorithm& trackAlg) 
{
    // Если новый объект
    if (id != idSave) 
    {
        // Проверяем содержимое массива ID для первой камеры и увеличиваем количество одинаковых объектов
        countSame = SameObject(arrayHist, countSame, idSave);

        if (trackAlg.report) 
        {
            // Проверяем содержимое массива ID для второй камеры и увеличиваем количество одинаковых объектов
            countSame = SameObject(arrayHist, countSame, trackAlg.idSave, trackAlg.arrayID);
        }
        // Обновляем счетчик кадров для гистограмм и для детектора
        countFrameCam = COUNT_FRAME;
    }
}


/*
 метод для определения нового объекта на полученном кадре
 Input:
 idsBoxes - координаты бокса и id нового объекта, numCam - номер камеры, frame - сам кадр, countSame - количество одинаковых объектов, 
 vectorHist - сохраненные гистограммы, tracker - трекеры объектов, trackAlg - объект класса алгоритма с другой камеры
 Output:
 верно определенный id объекта и бокс на кадре
*/
void TrackingAlgorithm::CameraTracking(std::vector<std::vector<int>> idsBoxes, int numCam, cv::Mat frame, int& countSame,
    std::vector<cv::Mat>& vectorHist, std::vector<EuclideanDistTracker>& tracker, TrackingAlgorithm& trackAlg)
{
    auto numCam2 = (numCam + 1) % 2;

    for (std::vector<int>& box_id : idsBoxes) 
    {
        auto x = box_id[0];
        auto y = box_id[1];
        auto w = box_id[2];
        auto h = box_id[3];
        auto id = box_id[4];

        auto y1 = IfBorder(y, h);
        report = false;
        cv::Mat framePlt = frame(cv::Rect(x, y, w, h));

        NewObject(id, numCam, numCam2, countSame, vectorHist, trackAlg);
        auto idGlobal = id - countSame;
        idSave = id;
        if (countFrameCam > 0) 
        {
            CallHistogram(framePlt, vectorHist, idGlobal, OPT_PARAM[numCam], arrayID);
            countFrameCam -= 1;
            idCorrect = idGlobal;
        }
        else 
        {
            idCorrect = FindMaxSameId(idGlobal);
            id = idCorrect;
            putText(frame, "Object " + std::to_string(id), cv::Point(x, y1), cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar(0, 0, 255), 2);
        }

        cv::rectangle(frame, cv::Point(x, y), cv::Point(x + w, y + h), cv::Scalar(255, 0, 0), 3);
        tracker[numCam2].idCount = tracker[numCam].idCount;
    }
}


/*
 метод для обновления трекера, получения гистограмм, поиск сравнения, выполнение алгоритма
 Input:
 detections - координаты бокса , numCam - номер камеры, frame -  кадр, countSame - количество одинаковых объектов, vectorHist - сохраненные гистограммы , tracker - трекеры объектов, trackAlg - объект класса алгоритма с другой камеры
 Output:
 бокс с верно определнным айди на кадре
*/
void TrackingAlgorithm::updateCameraTracking(std::vector<cv::Rect> detections, int numCam, cv::Mat frame, int& countSame, std::vector<cv::Mat>& vectorHist, std::vector<EuclideanDistTracker>& tracker, TrackingAlgorithm& trackAlg)
{
    report = true;
    // Обновление трекера
    std::vector<std::vector<int>> idsBoxes1 = tracker[numCam].update(detections);
    // Трекинг, назначение ID, отрисовка боксов
    CameraTracking(idsBoxes1, numCam, frame, countSame, vectorHist,  tracker, trackAlg);
}