#include "object_detect.h"

static constexpr char model_path[] = "./ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite";
static constexpr char labels_path[] = "./coco_labels.txt";

ObjectDetect::ObjectDetect() : TFLiteModel(model_path, labels_path) {
    const std::vector<int> &outputs = interpreter->outputs();
    TfLiteTensor *boxes_tensor = interpreter->tensor(outputs[0]);
    TfLiteTensor *classes_tensor = interpreter->tensor(outputs[1]);
    TfLiteTensor *scores_tensor = interpreter->tensor(outputs[2]);
    TfLiteTensor *count_tensor = interpreter->tensor(outputs[3]);

    TfLiteIntArray* boxes_dims = boxes_tensor->dims; // float, [1, 20, 4]
    TfLiteIntArray* classes_dims = classes_tensor->dims; // float, [1, 20]
    TfLiteIntArray* scores_dims = scores_tensor->dims; // float, [1, 20]
    TfLiteIntArray* count_dims = count_tensor->dims; // float, [1]

    boxes_data = reinterpret_cast<box_coord *>(interpreter->typed_output_tensor<float>(0));
    classes_data = interpreter->typed_output_tensor<float>(1);
    scores_data = interpreter->typed_output_tensor<float>(2);
    count_data = interpreter->typed_output_tensor<float>(3);
}

void ObjectDetect::process_result(std::shared_ptr<cv::Mat> &frame) {
    float width = frame->size().width;
    float height = frame->size().height;
    cv::Scalar green(0, 255, 0);
    cv::Scalar black(0, 0, 0);

    for (int i = 0, count = (int) *count_data; i < count; ++i) {
        const int class_id = (int) classes_data[i];
        const float score = scores_data[i];
        if (class_id >= 0 && class_id < labels.size() && score >= 0.5f) {
            const box_coord &box = boxes_data[i];
            const std::string &label = labels[class_id];
            cv::Point topLeft(box.left * width, box.top * height);
            cv::Point bottomRight(box.right * width, box.bottom * height);
            cv::rectangle(*frame, topLeft, bottomRight, green, 1);
            cv::rectangle(*frame, cv::Point(topLeft.x, topLeft.y - 40), cv::Point(bottomRight.x, topLeft.y), green, cv::FILLED);
            cv::putText(*frame, label, cv::Point(topLeft.x + 10, topLeft.y - 15), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, black, 1, cv::LINE_AA);
        }
    }
}
