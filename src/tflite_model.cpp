#include "tflite_model.h"

#include "tensorflow/lite/shared_library.h"

#ifdef PROFILE
#include <chrono>
#endif

#include <fstream>

static std::vector<std::string> read_labels(const char* file_name)
{
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

bool ExternalLib::load(const char* lib_path)
{
    void* handle = tflite::SharedLibrary::LoadLibrary(lib_path);
    if (handle == nullptr) {
        std::cerr << "Unable to load external delegate from : " << lib_path << std::endl;
    } else {
        create = reinterpret_cast<decltype(create)>(
            tflite::SharedLibrary::GetLibrarySymbol(handle, "tflite_plugin_create_delegate"));
        destroy = reinterpret_cast<decltype(destroy)>(
            tflite::SharedLibrary::GetLibrarySymbol(handle, "tflite_plugin_destroy_delegate"));
        return create && destroy;
    }
    return false;
}

TFLiteModel::TFLiteModel(const char* model_path, const char* labels_path)
{
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

    if (posenet_lib.load("posenet_decoder.so")) {
        posenet_delegate = posenet_lib.create(nullptr, nullptr, 0, nullptr);
        if (posenet_delegate != nullptr) {
            interpreter->ModifyGraphWithDelegate(posenet_delegate);
            std::cout << "PoseNet delegate added" << std::endl;
        }
    }

    // Bind given context with interpreter
    interpreter->SetExternalContext(kTfLiteEdgeTpuContext, context.get());
    interpreter->SetNumThreads(1);
    if (interpreter->AllocateTensors() != kTfLiteOk) {
        std::cerr << "Failed to allocate tensors." << std::endl;
        exit(-1);
    }

    const std::vector<int>& inputs = interpreter->inputs();
    for (int input : inputs) {
        TfLiteTensor* input_tensor = interpreter->tensor(input);
        std::cout << "Input: " << input_tensor->name << " (type " << input_tensor->type << ") [";
        TfLiteIntArray* input_dims = input_tensor->dims;
        for (int i = 0; i < input_dims->size; ++i) {
            if (i > 0)
                std::cout << ", ";
            std::cout << input_tensor->dims->data[i];
        }
        std::cout << "]" << std::endl;
    }

    const std::vector<int>& outputs = interpreter->outputs();
    for (int output : outputs) {
        TfLiteTensor* output_tensor = interpreter->tensor(output);
        std::cout << "Output: " << output_tensor->name << " (type " << output_tensor->type << ") [";
        TfLiteIntArray* output_dims = output_tensor->dims;
        for (int i = 0; i < output_dims->size; ++i) {
            if (i > 0)
                std::cout << ", ";
            std::cout << output_tensor->dims->data[i];
        }
        std::cout << "]" << std::endl;
    }

    // Get input dimension from the input tensor metadata, assuming one input only
    TfLiteTensor* input_tensor = interpreter->tensor(interpreter->inputs()[0]);
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

TFLiteModel::~TFLiteModel()
{
    if (posenet_delegate != nullptr) {
        posenet_lib.destroy(posenet_delegate);
    }
}

#ifdef PROFILE
using hires_clock = std::chrono::high_resolution_clock;
#endif

void TFLiteModel::process_frame(std::shared_ptr<cv::Mat>& frame)
{
#ifdef PROFILE
    hires_clock::time_point start = hires_clock::now();
#endif

    cv::resize(*frame, input_image, input_image.size());
    cv::cvtColor(input_image, input_image, cv::COLOR_BGR2RGB);

#ifdef PROFILE
    hires_clock::time_point middle = hires_clock::now();
#endif

    if (interpreter->Invoke() != kTfLiteOk) {
        std::cerr << "Failed to invoke tflite!";
        exit(-1);
    }

#ifdef PROFILE
    hires_clock::time_point finish = hires_clock::now();
    auto resize_duration
        = std::chrono::duration_cast<std::chrono::nanoseconds>(middle - start).count();
    auto invoke_duration
        = std::chrono::duration_cast<std::chrono::nanoseconds>(finish - middle).count();
    std::cout << "Resize: " << (resize_duration / 1000000) << " ms" << std::endl;
    std::cout << "Invoke: " << (invoke_duration / 1000000) << " ms" << std::endl;
#endif

    process_result(frame);
}
