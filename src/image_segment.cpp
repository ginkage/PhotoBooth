#include "image_segment.h"

static constexpr char model_path[] = "./deeplabv3_mnv2_pascal_quant_edgetpu.tflite";
static constexpr char labels_path[] = "./pascal_voc_segmentation_labels.txt";

ImageSegment::ImageSegment() : TFLiteModel(model_path, labels_path) {
    const std::vector<int> &outputs = interpreter->outputs();
    TfLiteTensor *output_tensor = interpreter->tensor(outputs[0]);
    TfLiteIntArray* output_dims = output_tensor->dims; // int64, [1, 513, 513]

    pixels = interpreter->typed_output_tensor<int64_t>(0);
    output_height = output_dims->data[1];
    output_width = output_dims->data[2];
}

void ImageSegment::process_result(std::shared_ptr<cv::Mat> &frame) {
}
