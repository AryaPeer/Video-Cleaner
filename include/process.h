#pragma once

#include <string>
#include <memory>
#include <opencv2/opencv.hpp>

#include "filters.h"

// Forward declaration
class VideoDenoiser;

/**
 * Handles the video processing pipeline
 */
class VideoProcessor {
public:
    /**
     * Constructor
     * @param lowCutoff Lower cutoff frequency in Hz for audio filter
     * @param highCutoff Higher cutoff frequency in Hz for audio filter
     * @param noiseReduction Audio noise reduction factor (0-1)
     * @param videoDenoiseStrength Video denoising strength (0-100)
     */
    VideoProcessor(float lowCutoff, float highCutoff, float noiseReduction, float videoDenoiseStrength);
    
    /**
     * Processes a video file
     * @param inputPath Input video path
     * @param outputPath Output video path
     * @return True if successful
     */
    bool processVideo(const std::string& inputPath, const std::string& outputPath);

private:
    float m_lowCutoff;
    float m_highCutoff;
    float m_noiseReduction;
    float m_videoDenoiseStrength;

    std::unique_ptr<AudioProcessor> m_audioProcessor;
    std::unique_ptr<VideoDenoiser> m_videoDenoiser;

    bool extractAudio(const std::string& videoPath, std::vector<float>& audioData, int& sampleRate, int& channels);
    bool processAudio(std::vector<float>& audioData, int sampleRate, int channels);
    bool saveProcessedAudioToWav(const std::string& wavPath, const std::vector<float>& audioData,
                               int audioSampleRate, int audioChannels);
    bool processVideoFrames(const std::string& inputPath, const std::string& outputPath,
                           const std::vector<float>& processedAudio, int audioSampleRate, int audioChannels);

    cv::Mat denoiseFrame(const cv::Mat& frame);
    void applyAdditionalVideoEnhancements(cv::Mat& frame);
};