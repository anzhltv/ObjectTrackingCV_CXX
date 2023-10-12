#pragma once
#include <cstdio>
#include <cmath>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <inference_engine.hpp>

using namespace std;
using namespace InferenceEngine;

// Класс инициализации нейронной сети
class NeuralNetworkDetector {
public:
        /*
        загрузка модели нейроной сети
        input
        model_format - формат модели, size - размер входных данных, device -  устройство
        output
        уменьшенное изображения для подачи на вход сети
        */
        NeuralNetworkDetector(const string & model_format, const string & model_path, const cv::Size & size, const string & device = "CPU") {
                ie = InferenceEngine::Core();
                if (model_format == "openvino") {
                    net = ie.ReadNetwork(model_path + ".xml", model_path + ".bin");
                }
                else if (model_format == "onnx") {
                    net = ie.ReadNetwork(model_path + ".onnx");
                }
                exec_net = ie.LoadNetwork(net, device);
                input_size = size;
        }


        /*
        подготовка входного изображения
        input
        frame - изображение
        output
        уменьшенное изображения для подачи на вход сети
        */
        cv::Mat data_preparation(const cv::Mat& frame);


        /*
        обработка данных с изображения нейронной сетью
        input
        inputFrame - подготовленное изображение
        output
        выходные данные сети
        */
        Blob::Ptr forward(const cv::Mat& input_frame);

private:
    InferenceEngine::Core ie;
    InferenceEngine::CNNNetwork net;
    InferenceEngine::ExecutableNetwork exec_net;
    cv::Size input_size;

    /*
    перевод данных из Mat в Blob
    input
    mat - изображение
    output
    данные в формате blob c precision FP32
    */
    InferenceEngine::Blob::Ptr wrapMat2Blob(const cv::Mat& mat);
};