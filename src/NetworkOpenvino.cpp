#include "NetworkOpenvino.h";


/*
подготовка входного изображения 
input 
frame - изображение
output
уменьшенное изображения для подачи на вход сети
*/
auto NeuralNetworkDetector::data_preparation(const cv::Mat& frame) {
    // Изменяем размер изображения до размера входа сети
    cv::Mat resized_frame;
    cv::resize(frame, resized_frame, input_size);

    // Преобразуем изображение в нужный формат и порядок каналов
    cv::Mat input_frame;
    cv::cvtColor(resized_frame, input_frame, cv::COLOR_BGR2RGB); // Преобразуем BGR в RGB
    resized_frame.convertTo(resized_frame, CV_32F); // Преобразуем к типу float32
    //cv::imshow("input", input_frame);
    return resized_frame;
}


/*
обработка данных с изображения нейронной сетью
input
inputFrame - подготовленное изображение
output
выходные данные сети
*/
Blob::Ptr NeuralNetworkDetector::forward(const cv::Mat& inputFrame) {
    // Получаем информацию о входных данных
    ConstInputsDataMap inputsInfo = exec_net.GetInputsInfo();
    if (inputsInfo.size() != 1) {
        throw std::logic_error("Expected exactly one input blob");
    }

    // Получаем имя входного блоа
    std::string input_name = inputsInfo.begin()->first;

    // Создаем InferRequest
    InferRequest inferRequest = exec_net.CreateInferRequest();
    try {
        // Подготавливаем входные данные
        Blob::Ptr inputBlob = wrapMat2Blob(inputFrame);
        try
        {
            // Устанавливаем входные данные в InferRequest
            inferRequest.SetBlob(input_name, inputBlob);
            try
            {
                // Вызов инференса
                inferRequest.Infer();
                // Получаем результаты инференса
                ConstOutputsDataMap outputsInfo = exec_net.GetOutputsInfo();
                if (outputsInfo.size() != 1)
                {
                    throw std::logic_error("Expected exactly one output blob");
                }

                // Получаем имя выходного
                std::string output_name = outputsInfo.begin()->first;

                // Получаем выходной блоа из InferRequest
                Blob::Ptr outputBlob = inferRequest.GetBlob(output_name);

                return outputBlob;
            }
            catch (const InferenceEngine::Exception& e)
            {
                // Обработка исключения, возникшего во время инференса
                std::cerr << "Infer error: " << e.what() << std::endl;
            }
        }
        catch (const InferenceEngine::Exception& e)
        {
            // Обработка исключения, возникшего во время инференса
            std::cerr << "SetBlob error: " << e.what() << std::endl;
        }
    }
    catch (const InferenceEngine::Exception& e)
    {
        // Обработка исключения, возникшего во время инференса

        std::cerr << "wrapMat2Blob error: " << e.what() << std::endl;
        Blob::Ptr outputBlob;
        return outputBlob;
    }
}


/*
перевод данных из Mat в Blob
input
mat - изображение
output
данные в формате blob c precision FP32
*/
InferenceEngine::Blob::Ptr NeuralNetworkDetector::wrapMat2Blob(const cv::Mat& mat)
{
    InferenceEngine::TensorDesc tensorDesc(InferenceEngine::Precision::FP32,
        { 1, static_cast<size_t>(mat.channels()), static_cast<size_t>(mat.rows), static_cast<size_t>(mat.cols) },
        InferenceEngine::Layout::NCHW);
    return InferenceEngine::make_shared_blob<float>(tensorDesc, (float*)(mat.data), mat.total() * 3);

}