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
    pixel_count = output_height * (size_t) output_width;

    mask = cv::Mat(output_height, output_width, CV_8UC3);

    for (int i = 1; i < 256; ++i)
        for (int j = 0; j < 3; ++j)
            colors[i][j] = (colors[i - 1][j] + rand() % 256) / 2;

    std::cout << sizeof(colors[0]) << std::endl;
}

void ImageSegment::process_result(std::shared_ptr<cv::Mat> &frame) {
    if (big_mask.total() != frame->total())
        big_mask = cv::Mat(frame->size(), frame->type());

    cv::Vec3b *pmask = (cv::Vec3b *) mask.ptr();
    int64_t *ppx = pixels;
    for (size_t i = 0; i < pixel_count; ++i)
        *pmask++ = colors[*ppx++];

    cv::resize(mask, big_mask, big_mask.size());
    cv::add(*frame, big_mask, *frame);
}

