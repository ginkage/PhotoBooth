#pragma once

#include <vector>

#include <opencv2/opencv.hpp>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "edgetpu.h"

class TFLiteModel {
public:
    TFLiteModel(const char *model_path, const char *labels_path);
    void invoke(std::shared_ptr<cv::Mat> &frame);

protected:
    // Common stuff for TFLite models
    std::vector<std::string> labels;
    std::unique_ptr<tflite::FlatBufferModel> model;
    std::shared_ptr<edgetpu::EdgeTpuContext> context;
    tflite::ops::builtin::BuiltinOpResolver resolver;
    std::unique_ptr<tflite::Interpreter> interpreter;
    cv::Mat input_image;
};

