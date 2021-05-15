template <class T>
float confidence(T prediction) {
    return (float)prediction;
}

template <>
float confidence<int8_t>(int8_t prediction) {
    return ((float)prediction + 128.0f) / 255.0f;
}

template <>
float confidence<uint8_t>(uint8_t prediction) {
    return (float)prediction / 255.0f;
}

template <class T>
void get_top_n(const T* predictions, int predictions_size, std::vector<size_t> &indexes,
        std::vector<std::pair<size_t, float>> &result)
{
    std::iota(indexes.begin(), indexes.end(), 0);

    // Sort indexes based on comparing prediction values
    std::partial_sort(indexes.begin(), indexes.begin() + result.size(), indexes.end(),
            [&predictions](size_t index1, size_t index2) {
                return predictions[index1] > predictions[index2];
            });

    // Make a std::pair and append it to the result for N predictions
    for (size_t i = 0; i < result.size(); ++i) {
         result[i] = std::make_pair(indexes[i], confidence(predictions[indexes[i]]));
    }
}

inline void get_top_n(const tflite::Interpreter* interpreter, int output_type, int output_size,
        std::vector<size_t> &indexes, std::vector<std::pair<size_t, float>> &result) {
    switch (output_type) {
        case kTfLiteFloat32:
            get_top_n<float>(interpreter->typed_output_tensor<float>(0), output_size, indexes, result);
            break;
        case kTfLiteInt8:
            get_top_n<int8_t>(interpreter->typed_output_tensor<int8_t>(0), output_size, indexes, result);
            break;
        case kTfLiteUInt8:
            get_top_n<uint8_t>(interpreter->typed_output_tensor<uint8_t>(0), output_size, indexes, result);
            break;
        default:
          std::cerr << "Cannot handle output type " << output_type << std::endl;
          exit(-1);
    }
}

