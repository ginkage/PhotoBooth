#include <opencv2/opencv.hpp>
#include <thread>
#include <numeric>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <vector>

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"

#include "edgetpu.h"

#include "thread_sync.h"
#include "fps.h"

static std::shared_ptr<cv::Mat> frame_to_process;
static std::shared_ptr<cv::Mat> frame_to_display;
static ThreadSync process_sync;
static ThreadSync display_sync;
static bool terminate = false;
static constexpr char win[] = "PhotoBooth";
static constexpr char model_path[] = "./ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite";
static constexpr char labels_path[] = "./coco_labels.txt";

std::vector<std::string> read_labels(const char *file_name) {
    std::ifstream file(file_name);
    if (!file) {
        std::cerr << "Labels file " << file_name << " not found";
        return {};
    }

    std::vector<std::string> result;
    std::string line;
    while (std::getline(file, line)) {
        result.push_back(line);
    }

    return result;
}

void capture_thread() {
    // Request MJPEG, 1280x720, 30fps
    cv::VideoCapture camera("v4l2src ! image/jpeg,width=1280,height=720,framerate=30/1 ! jpegdec ! videoconvert ! video/x-raw,format=BGR ! appsink");

    while (!terminate) {
        // Keep 'em coming
        auto frame = std::make_shared<cv::Mat>();
        camera >> *frame;
        process_sync.produce([&] { frame_to_process = frame; });
    }

    // Make sure that process thread is unblocked
    process_sync.produce([] {});
}

struct box_coord {
    float top;
    float left;
    float bottom;
    float right;
};

void process_thread() {
    std::vector<std::string> labels = read_labels(labels_path);

    // Load model
    std::unique_ptr<tflite::FlatBufferModel> model =
            tflite::FlatBufferModel::BuildFromFile(model_path);

    // Create context for use with Coral
    std::shared_ptr<edgetpu::EdgeTpuContext> context =
            edgetpu::EdgeTpuManager::GetSingleton()->OpenDevice();

    // Create model interpreter
    tflite::ops::builtin::BuiltinOpResolver resolver;
    resolver.AddCustom(edgetpu::kCustomOp, edgetpu::RegisterCustomOp());
    std::unique_ptr<tflite::Interpreter> interpreter;
    if (tflite::InterpreterBuilder(*model, resolver)(&interpreter) != kTfLiteOk) {
        std::cerr << "Failed to build interpreter." << std::endl;
        exit(-1);
    }

    // Bind given context with interpreter
    interpreter->SetExternalContext(kTfLiteEdgeTpuContext, context.get());
    interpreter->SetNumThreads(1);
    if (interpreter->AllocateTensors() != kTfLiteOk) {
        std::cerr << "Failed to allocate tensors." << std::endl;
        exit(-1);
    }

    // Get input dimension from the input tensor metadata, assuming one input only
    TfLiteTensor *input_tensor = interpreter->tensor(interpreter->inputs()[0]);
    TfLiteType input_type = input_tensor->type;
    TfLiteIntArray* input_dims = input_tensor->dims;
    int input_size = input_tensor->bytes;
    int wanted_height = input_dims->data[1];
    int wanted_width = input_dims->data[2];
    int wanted_channels = input_dims->data[3];

    std::cout << "Input width: " << wanted_width << ", height: " << wanted_height
            << ", channels: " << wanted_channels << ", type: " << input_type << std::endl;

    uint8_t* input_data = interpreter->typed_input_tensor<uint8_t>(0);

    const std::vector<int> &outputs = interpreter->outputs();
    TfLiteTensor *boxes_tensor = interpreter->tensor(outputs[0]);
    TfLiteTensor *classes_tensor = interpreter->tensor(outputs[1]);
    TfLiteTensor *scores_tensor = interpreter->tensor(outputs[2]);
    TfLiteTensor *count_tensor = interpreter->tensor(outputs[3]);

    TfLiteIntArray* boxes_dims = boxes_tensor->dims; // float, [1, 20, 4]
    TfLiteIntArray* classes_dims = classes_tensor->dims; // float, [1, 20]
    TfLiteIntArray* scores_dims = scores_tensor->dims; // float, [1, 20]
    TfLiteIntArray* count_dims = count_tensor->dims; // float, [1]

    float* boxes_data = interpreter->typed_output_tensor<float>(0);
    float* classes_data = interpreter->typed_output_tensor<float>(1);
    float* scores_data = interpreter->typed_output_tensor<float>(2);
    float* count_data = interpreter->typed_output_tensor<float>(3);
    box_coord *boxes = reinterpret_cast<box_coord *>(boxes_data);

    std::shared_ptr<cv::Mat> frame;
    cv::Mat input_image(wanted_width, wanted_height, CV_8UC3, input_data);

    while (!terminate) {
        process_sync.consume(
            [&] {
                // Check if there's a new frame available to process
                return terminate || frame != frame_to_process;
            },
            [&] {
                // Make sure it doesn't get garbage-collected
                if (!terminate && frame != frame_to_process) {
                    frame = frame_to_process;
                }
            },
            [&] {
                // Process the frame
                if (!terminate && frame) {
                    cv::resize(*frame, input_image, input_image.size());
                    cv::cvtColor(input_image, input_image, cv::COLOR_BGR2RGB);
                    if (interpreter->Invoke() != kTfLiteOk) {
                        std::cerr << "Failed to invoke tflite!";
                        exit(-1);
                    }

                    float width = frame->size().width;
                    float height = frame->size().height;

                    for (int i = 0, count = (int) *count_data; i < count; ++i) {
                        const int class_id = (int) classes_data[i];
                        const float score = scores_data[i];
                        if (class_id >= 0 && class_id < labels.size() && score >= 0.5f) {
                            const box_coord &box = boxes[i];
                            const std::string &label = labels[class_id];
                            cv::Point topLeft(box.left * width, box.top * height);
                            cv::Point bottomRight(box.right * width, box.bottom * height);
                            cv::rectangle(*frame, topLeft, bottomRight, cv::Scalar(0, 255, 0), 1);
                            cv::rectangle(*frame, cv::Point(topLeft.x, topLeft.y - 40), cv::Point(bottomRight.x, topLeft.y), cv::Scalar(0, 255, 0), cv::FILLED);
                            cv::putText(*frame, label, cv::Point(topLeft.x + 10, topLeft.y - 15), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
                        }
                    }

                    display_sync.produce([&] { frame_to_display = frame; });
                }
            });
    }

    // Make sure that display thread is unblocked
    display_sync.produce([] {});
}

void display_thread() {
    Fps fps;
    std::shared_ptr<cv::Mat> frame;
    while (!terminate) {
        display_sync.consume(
            [&] {
                // Check if there's a new frame available to display
                return terminate || frame != frame_to_display;
            },
            [&] {
                // Make sure it doesn't get garbage-collected
                if (!terminate && frame != frame_to_display) {
                    frame = frame_to_display;
                }
            },
            [&] {
                // Display the frame
                if (!terminate && frame) {
                    cv::imshow(win, *frame);
                    fps.tick(60);
                }
            });
    }
}

int main(int argc __attribute__((unused)), char** argv __attribute__((unused)))
{
    // Create the window in advance
    cv::namedWindow(win);

    std::thread capture([] { capture_thread(); });
    std::thread process([] { process_thread(); });
    std::thread display([] { display_thread(); });

    // Wait for the Escape key
    while ((cv::waitKey(1) & 0xFF) != 27);

    terminate = true;
    capture.join();
    process.join();
    display.join();

    return 0;
}
