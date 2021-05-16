#pragma once

#include "tflite_model.h"

class ImageSegment : public TFLiteModel {
public:
    ImageSegment();

protected:
    void process_result(std::shared_ptr<cv::Mat> &frame) override;

private:
    int64_t* pixels;
    int output_width;
    int output_height;
};
