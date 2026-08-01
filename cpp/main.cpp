#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>

int main() {
    // Load ONNX model
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "EmotionRecognition");
    Ort::SessionOptions session_options;
    Ort::Session session(env, "fer2013_vit.onnx", session_options);

    // Labels
    std::vector<std::string> labels = {"Angry","Disgust","Fear","Happy","Sad","Surprise","Neutral"};

    // Load video
    cv::VideoCapture cap("./video_samples/video_sample_1.mp4");
    cv::VideoWriter writer("./output/video_sample_1_output.mp4",
                           cv::VideoWriter::fourcc('m','p','4','v'),   
                           cap.get(cv::CAP_PROP_FPS),
                           cv::Size((int)cap.get(cv::CAP_PROP_FRAME_WIDTH),
                                    (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT)));

    cv::CascadeClassifier face_cascade("./haarcascade_frontalface_default.xml");

    std::map<std::string,double> totals;
    for (auto &l : labels) totals[l] = 0.0;

    cv::Mat frame;
    while (cap.read(frame)) {
        std::vector<cv::Rect> faces;
        face_cascade.detectMultiScale(frame, faces);

        for (auto &face : faces) {
            cv::Mat roi = frame(face);
            cv::resize(roi, roi, cv::Size(224,224));
            cv::cvtColor(roi, roi, cv::COLOR_BGR2RGB);

            // Convert to float tensor
            roi.convertTo(roi, CV_32F, 1.0/255.0);

            // flatten to NCHW format
            std::vector<float> inputTensorValues;
            inputTensorValues.reserve(3 * 224 * 224);
            for (int c = 0; c < 3; c++) {
                for (int y = 0; y < 224; y++) {
                    for (int x = 0; x < 224; x++) {
                        inputTensorValues.push_back(roi.at<cv::Vec3f>(y,x)[c]);
                    }
                }
            }

            // Create tensor
            std::array<int64_t,4> inputShape{1,3,224,224};
            Ort::MemoryInfo memInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
            Ort::Value inputTensor = Ort::Value::CreateTensor<float>(memInfo,
                inputTensorValues.data(), inputTensorValues.size(), inputShape.data(), inputShape.size());

            // Run inference
            const char* input_names[] = {"input"};
            const char* output_names[] = {"output"};

            auto output = session.Run(Ort::RunOptions{nullptr},
                          input_names, &inputTensor, 1,
                          output_names, 1);

            float* scores = output.front().GetTensorMutableData<float>();

            
            // Apply softmax to convert logits into probabilities
            std::vector<float> probs(labels.size());
            float maxLogit = *std::max_element(scores, scores + labels.size());

            // numerical stability trick
            float sumExp = 0.0f;
            for (int i = 0; i < labels.size(); i++) {
                probs[i] = std::exp(scores[i] - maxLogit);
                sumExp += probs[i];
            }
            for (int i = 0; i < labels.size(); i++) {
                probs[i] /= sumExp;
            }

            // Debug printout of probabilities
            std::cout << "Scores (softmax): ";
            for (int i = 0; i < labels.size(); i++) {
                std::cout << labels[i] << "=" << probs[i] << " ";
            }
            std::cout << std::endl;

            // Update totals  
            int topIdx = 0;
            float maxScore = probs[0];
            for (int i=0; i<labels.size(); i++) {
                totals[labels[i]] += probs[i];
                if (probs[i] > maxScore) { maxScore = probs[i]; topIdx = i; }
            }

            // Draw
            cv::rectangle(frame, face, cv::Scalar(255,0,0), 2);
            cv::putText(frame, labels[topIdx], cv::Point(face.x, face.y-10),
                        cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0,255,0), 2);
        }
        writer.write(frame);
    }

    cap.release();
    writer.release();

    // Print totals
    std::cout << "Output:\n";
    double grandTotal = 0.0;
    for (auto &kv : totals) {
        grandTotal += kv.second;
    }
    
    // Normalize totals into percentages
    int idx = 0;
    for (auto &kv : totals) {
        double percent = (grandTotal > 0.0) ? (kv.second / grandTotal * 100.0) : 0.0;
        std::cout << idx++ << " " << kv.first << " " << percent << "%\n";
    }
}