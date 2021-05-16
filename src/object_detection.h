#pragma once

#include "tflite_model.h"

struct box_coord {
    float top;
    float left;
    float bottom;
    float right;
};

class ObjectDetection : public TFLiteModel {
public:
    ObjectDetection();
    void detect_objects(std::shared_ptr<cv::Mat> &frame);

private:
    // Object Detection specific
    box_coord *boxes_data;
    float* classes_data;
    float* scores_data;
    float* count_data;
};


