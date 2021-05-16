#pragma once

#include "tflite_model.h"

class ImageClassify : public TFLiteModel {
public:
    ImageClassify();

protected:
    void process_result(std::shared_ptr<cv::Mat> &frame) override;

private:
    float* scores;
    int scores_count;
    TfLiteType output_type;
    std::vector<size_t> indexes;
    std::vector<std::pair<size_t, float>> result;
};
