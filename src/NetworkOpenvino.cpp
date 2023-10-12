#include "NetworkOpenvino.h";


/*
���������� �������� �����������
input
frame - �����������
output
����������� ����������� ��� ������ �� ���� ����
*/
cv::Mat NeuralNetworkDetector::data_preparation(const cv::Mat& frame) {
    // �������� ������ ����������� �� ������� ����� ����
    cv::Mat resized_frame;
    cv::resize(frame, resized_frame, input_size);

    // ����������� ����������� � ������ ������ � ������� �������
    cv::Mat input_frame;
    cv::cvtColor(resized_frame, input_frame, cv::COLOR_BGR2RGB); // ����������� BGR � RGB
    resized_frame.convertTo(resized_frame, CV_32F); // ����������� � ���� float32
    //cv::imshow("input", input_frame);
    return resized_frame;
}


/*
��������� ������ � ����������� ��������� �����
input
inputFrame - �������������� �����������
output
�������� ������ ����
*/
Blob::Ptr NeuralNetworkDetector::forward(const cv::Mat& inputFrame) {
    // �������� ���������� � ������� ������
    ConstInputsDataMap inputsInfo = exec_net.GetInputsInfo();
    if (inputsInfo.size() != 1) {
        throw std::logic_error("Expected exactly one input blob");
    }

    // �������� ��� �������� ����
    std::string input_name = inputsInfo.begin()->first;

    // ������� InferRequest
    InferRequest inferRequest = exec_net.CreateInferRequest();
    try {
        // �������������� ������� ������
        Blob::Ptr inputBlob = wrapMat2Blob(inputFrame);
        try
        {
            // ������������� ������� ������ � InferRequest
            inferRequest.SetBlob(input_name, inputBlob);
            try
            {
                // ����� ���������
                inferRequest.Infer();
                // �������� ���������� ���������
                ConstOutputsDataMap outputsInfo = exec_net.GetOutputsInfo();
                if (outputsInfo.size() != 1)
                {
                    throw std::logic_error("Expected exactly one output blob");
                }

                // �������� ��� ���������
                std::string output_name = outputsInfo.begin()->first;

                // �������� �������� ���� �� InferRequest
                Blob::Ptr outputBlob = inferRequest.GetBlob(output_name);

                return outputBlob;
            }
            catch (const InferenceEngine::Exception& e)
            {
                // ��������� ����������, ���������� �� ����� ���������
                std::cerr << "Infer error: " << e.what() << std::endl;
            }
        }
        catch (const InferenceEngine::Exception& e)
        {
            // ��������� ����������, ���������� �� ����� ���������
            std::cerr << "SetBlob error: " << e.what() << std::endl;
        }
    }
    catch (const InferenceEngine::Exception& e)
    {
        // ��������� ����������, ���������� �� ����� ���������

        std::cerr << "wrapMat2Blob error: " << e.what() << std::endl;
        Blob::Ptr outputBlob;
        return outputBlob;
    }
}


/*
������� ������ �� Mat � Blob
input
mat - �����������
output
������ � ������� blob c precision FP32
*/
InferenceEngine::Blob::Ptr NeuralNetworkDetector::wrapMat2Blob(const cv::Mat& mat)
{
    InferenceEngine::TensorDesc tensorDesc(InferenceEngine::Precision::FP32,
        { 1, static_cast<size_t>(mat.channels()), static_cast<size_t>(mat.rows), static_cast<size_t>(mat.cols) },
        InferenceEngine::Layout::NCHW);
    return InferenceEngine::make_shared_blob<float>(tensorDesc, (float*)(mat.data), mat.total() * 3);

}