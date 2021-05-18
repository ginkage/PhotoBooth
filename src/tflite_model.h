#pragma once

#include <opencv2/opencv.hpp>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "edgetpu.h"

#include <vector>

struct ExternalLib {
    using CreateDelegatePtr = std::add_pointer<TfLiteDelegate*(
        const char**, const char**, size_t,
        void (*report_error)(const char*))>::type;
    using DestroyDelegatePtr = std::add_pointer<void(TfLiteDelegate*)>::type;

    bool load(const char *lib_path);

    CreateDelegatePtr create{nullptr};
    DestroyDelegatePtr destroy{nullptr};
};

class TFLiteModel {
public:
    TFLiteModel(const char *model_path, const char *labels_path);
    ~TFLiteModel();

    void process_frame(std::shared_ptr<cv::Mat> &frame);

protected:
    virtual void process_result(std::shared_ptr<cv::Mat> &frame) = 0;

protected:
    std::vector<std::string> labels;
    std::unique_ptr<tflite::FlatBufferModel> model;
    std::shared_ptr<edgetpu::EdgeTpuContext> context;
    tflite::ops::builtin::BuiltinOpResolver resolver;
    std::unique_ptr<tflite::Interpreter> interpreter;
    cv::Mat input_image;
    TfLiteDelegate *posenet_delegate;
    ExternalLib posenet_lib;
};
