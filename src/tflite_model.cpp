#include "tflite_model.h"

#include <fstream>

static std::vector<std::string> read_labels(const char *file_name) {
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

TFLiteModel::TFLiteModel(const char *model_path, const char *labels_path) {
    labels = read_labels(labels_path);

    // Load model
    model = tflite::FlatBufferModel::BuildFromFile(model_path);

    // Create context for use with Coral
    context = edgetpu::EdgeTpuManager::GetSingleton()->OpenDevice();

    // Create model interpreter
    resolver.AddCustom(edgetpu::kCustomOp, edgetpu::RegisterCustomOp());
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
    input_image = cv::Mat(wanted_width, wanted_height, CV_8UC3, input_data);
}

void TFLiteModel::invoke(std::shared_ptr<cv::Mat> &frame) {
    cv::resize(*frame, input_image, input_image.size());
    cv::cvtColor(input_image, input_image, cv::COLOR_BGR2RGB);
    if (interpreter->Invoke() != kTfLiteOk) {
        std::cerr << "Failed to invoke tflite!";
        exit(-1);
    }
}

