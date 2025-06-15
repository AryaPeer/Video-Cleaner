#pragma once

#include <opencv2/opencv.hpp>

/**
 * Interface for video denoising
 */
class VideoDenoiser {
public:
    /**
     * Constructor
     * @param strength Denoising strength
     */
    VideoDenoiser(float strength) : m_strength(strength) {}
    
    /**
     * Destructor
     */
    virtual ~VideoDenoiser() {}

    /**
     * Initializes denoiser resources
     * @param width Frame width
     * @param height Frame height
     */
    virtual void initialize(int width, int height) = 0;

    /**
     * Denoises a frame
     * @param inputFrame Input frame
     * @return Denoised frame
     */
    virtual cv::Mat denoise(const cv::Mat& inputFrame) = 0;

protected:
    float m_strength;
};

/**
 * CPU implementation of video denoising
 */
class CPUVideoDenoiser : public VideoDenoiser {
public:
    /**
     * Constructor
     * @param strength Denoising strength
     */
    CPUVideoDenoiser(float strength);
    
    /**
     * Destructor
     */
    ~CPUVideoDenoiser() override;

    /**
     * Initializes CPU resources
     * @param width Frame width
     * @param height Frame height
     */
    void initialize(int width, int height) override;

    /**
     * Denoises a frame using CPU
     * @param inputFrame Input frame
     * @return Denoised frame
     */
    cv::Mat denoise(const cv::Mat& inputFrame) override;

private:
    int m_width = 0;
    int m_height = 0;
    bool m_initialized = false;
};

/**
 * Factory function to create video denoiser
 * @param strength Denoising strength
 * @return Video denoiser instance
 */
std::unique_ptr<VideoDenoiser> createVideoDenoiser(float strength);