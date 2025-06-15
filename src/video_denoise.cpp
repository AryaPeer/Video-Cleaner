#include "video_denoise.h"
#include <opencv2/opencv.hpp>
#include <iostream>

std::unique_ptr<VideoDenoiser> createVideoDenoiser(float strength) {
    return std::make_unique<CPUVideoDenoiser>(strength);
}

CPUVideoDenoiser::CPUVideoDenoiser(float strength)
    : VideoDenoiser(strength), m_initialized(false) {
}

CPUVideoDenoiser::~CPUVideoDenoiser() {
}

void CPUVideoDenoiser::initialize(int width, int height) {
    m_width = width;
    m_height = height;
    m_initialized = true;
}

cv::Mat CPUVideoDenoiser::denoise(const cv::Mat& inputFrame) {
    if (!m_initialized) {
        initialize(inputFrame.cols, inputFrame.rows);
    }

    cv::Mat result;

    if (m_strength < 33.0f) {
        cv::fastNlMeansDenoisingColored(inputFrame, result, 3.0f, 3.0f, 7, 21);
    } else if (m_strength < 66.0f) {
        cv::bilateralFilter(inputFrame, result, 9, 75, 75);
    } else {
        cv::Mat temp;
        cv::bilateralFilter(inputFrame, temp, 9, 100, 100);
        cv::fastNlMeansDenoisingColored(temp, result, 5.0f, 5.0f, 7, 35);
    }

    return result;
}