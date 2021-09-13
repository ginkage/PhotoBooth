#pragma once

#include "tflite_model.h"

struct box_coord {
    float top;
    float left;
    float bottom;
    float right;
};

class ObjectDetect : public TFLiteModel {
public:
    ObjectDetect();

protected:
    void process_result(std::shared_ptr<cv::Mat>& frame) override;

private:
    box_coord* boxes_data;
    float* classes_data;
    float* scores_data;
    float* count_data;
};
