#include "TrackingAlgorithm.h"
#include "SmallUtils.h"


constexpr auto COUNT_FRAME = 40; // ���������� ������ ��� ������������� ��������
const std::vector<float> OPT_PARAM = { 0.5f, 0.21f }; // ������� ��������� �������� ��� ������� �����
constexpr auto PART_FRAME = 6; // ����� �� ������ ����� ������ ��������


/* FindMaxSameId
 ����� ��� ����������� ����������� ���� ��� �������
 ���� ������������ ����� ���������� ������, ��� ���� �� 1/6 �� ���������� ������ ��������,
 �� ������ ������� ��������� ����,
 ����� ���� �� �������
 Input:
 idGlobal - ���� �������� �������
 Output:
 ���������� ���� �������
*/
auto TrackingAlgorithm::FindMaxSameId(int idGlobal) 
{
    std::vector<int> arrayID_list(arrayID.begin(), arrayID.end());
    // ���� ������������ ���������� ������ ��� 1/6 �� ������ ����� ������ ��������
    if (*std::max_element(arrayID_list.begin(), arrayID_list.end()) >= COUNT_FRAME / PART_FRAME)
    {
        // �� ����� ��������� id
        const auto idCorr = static_cast<int>(std::distance(arrayID_list.begin(), std::max_element(arrayID_list.begin(), arrayID_list.end())));
        return idCorr;
    }
    else 
    {
        // ����� ��������� �� �������
        const auto idCorr = idGlobal;
        return idCorr;
    }
}



/*
 ����� �� ������, ���� ������ ��� �� ������
 ���� ������������ ����� ���������� ������, ��� ���� �� 1/6 �� ���������� ������ ��������,
 �� ������� ������� ������� � ��������� �������� ��� ����� �������, ��� ��� �� ��� ����
 + ����������� ���������� ����������� ��������
 Input:
arrayHist - ������ � ������������ �������������, countSame - ������� ���������� ���������, idSave - ���� ����������� �������
 Output:
 ���������� ���������� ���������
 */
int TrackingAlgorithm::SameObject(std::vector<cv::Mat>& arrayHist, int countSame, int idSave) 
{
    //���� ��� ������ ������������ ������, �� ������� ���������� ������� � ������� id + ���������� ����� ����������� ��������
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
 ���������� ��� ������ �� ������, ���� ������ ��� �� ������, ��� ������ ������, � ���� ������ ����� ������ ���� �� ������� ������� ������
 ���� ������������ ����� ���������� ������, ��� ���� �� 1/6 �� ���������� ������ ��������,
 �� ������� ������� ������� � ��������� �������� ��� ����� �������, ��� ��� �� ��� ����
 + ����������� ���������� ����������� ��������
 Input:
arrayHist - ������ � ������������ �������������, countSame - ������� ���������� ���������, idSave - ���� ����������� �������
 Output:
 ���������� ���������� ���������
 */
int TrackingAlgorithm::SameObject(std::vector<cv::Mat>& arrayHist, int countSame, int idSave, std::vector<int>& arrID)
{
    //���� ��� ������ ������������ ������, �� ������� ���������� ������� � ������� id + ���������� ����� ����������� ��������
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
 ����� ��� ���������� countSame - ���� �������� ����� ������, � ������ ��� ��������� � ��� ������������
 Input:
 id �������� ������� �� �������, numCam, numCam2 - ����� ������� � ������ ������, countSame - ���������� ���������� ��������, arrayHist - ����������� �����������, trackAlg - ������ ������ ��������� � ������ ������
 Output:
 ����������� ������ arrayID
*/
void TrackingAlgorithm::NewObject(int id, int numCam, int numCam2, int& countSame, std::vector<cv::Mat>& arrayHist, TrackingAlgorithm& trackAlg) 
{
    // ���� ����� ������
    if (id != idSave) 
    {
        // ��������� ���������� ������� ID ��� ������ ������ � ����������� ���������� ���������� ��������
        countSame = SameObject(arrayHist, countSame, idSave);

        if (trackAlg.report) 
        {
            // ��������� ���������� ������� ID ��� ������ ������ � ����������� ���������� ���������� ��������
            countSame = SameObject(arrayHist, countSame, trackAlg.idSave, trackAlg.arrayID);
        }
        // ��������� ������� ������ ��� ���������� � ��� ���������
        countFrameCam = COUNT_FRAME;
    }
}


/*
 ����� ��� ����������� ������ ������� �� ���������� �����
 Input:
 idsBoxes - ���������� ����� � id ������ �������, numCam - ����� ������, frame - ��� ����, countSame - ���������� ���������� ��������, 
 vectorHist - ����������� �����������, tracker - ������� ��������, trackAlg - ������ ������ ��������� � ������ ������
 Output:
 ����� ������������ id ������� � ���� �� �����
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

        auto y1 = if_border(y, h);
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
 ����� ��� ���������� �������, ��������� ����������, ����� ���������, ���������� ���������
 Input:
 detections - ���������� ����� , numCam - ����� ������, frame -  ����, countSame - ���������� ���������� ��������, vectorHist - ����������� ����������� , tracker - ������� ��������, trackAlg - ������ ������ ��������� � ������ ������
 Output:
 ���� � ����� ����������� ���� �� �����
*/
void TrackingAlgorithm::updateCameraTracking(std::vector<cv::Rect> detections, int numCam, cv::Mat frame, int& countSame, std::vector<cv::Mat>& vectorHist, std::vector<EuclideanDistTracker>& tracker, TrackingAlgorithm& trackAlg)
{
    report = true;
    // ���������� �������
    std::vector<std::vector<int>> idsBoxes1 = tracker[numCam].update(detections);
    // �������, ���������� ID, ��������� ������
    CameraTracking(idsBoxes1, numCam, frame, countSame, vectorHist,  tracker, trackAlg);
}