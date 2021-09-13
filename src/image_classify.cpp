#include "image_classify.h"
#include "get_top_n.h"

#include <iostream>

static constexpr char model_path[] = "./tf2_mobilenet_v3_edgetpu_1.0_224_ptq_edgetpu.tflite";
static constexpr char labels_path[] = "./imagenet_labels.txt";

ImageClassify::ImageClassify()
    : TFLiteModel(model_path, labels_path)
{
    const std::vector<int>& outputs = interpreter->outputs();
    TfLiteTensor* output_tensor = interpreter->tensor(outputs[0]);
    TfLiteIntArray* output_dims = output_tensor->dims; // float, [1, 1, ..., size]

    scores = interpreter->typed_output_tensor<float>(0);
    scores_count = output_dims->data[output_dims->size - 1];
    output_type = output_tensor->type;
    indexes = std::vector<size_t>(scores_count);
    result = std::vector<std::pair<size_t, float>>(5);
}

void ImageClassify::process_result(std::shared_ptr<cv::Mat>& frame)
{
    get_top_n(interpreter.get(), output_type, scores_count, indexes, result);

    for (auto res : result) {
        if (res.second > 0.01f) {
            std::cout << labels[res.first] << " @ " << res.second << std::endl;
        }
    }
}
