#pragma once

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>

/**
 * Detects and extracts faces from videos
 */
class FaceExtractor {
public:
    /**
     * Constructor
     */
    FaceExtractor();
    
    /**
     * Extracts faces from a video at a specific timestamp
     * @param videoPath Path to the video file
     * @param timeInSeconds Timestamp in seconds to extract faces from
     * @param outputDir Directory where extracted faces will be saved
     * @return True if extraction was successful, false otherwise
     */
    bool extractFaces(const std::string& videoPath, float timeInSeconds, const std::string& outputDir);
    
    /**
     * Extracts faces from a video within a time range
     * @param videoPath Path to the video file
     * @param startTime Start time in seconds
     * @param endTime End time in seconds
     * @param interval Time interval in seconds between extractions
     * @param outputDir Directory where extracted faces will be saved
     * @return True if extraction was successful, false otherwise
     */
    bool extractFacesFromRange(const std::string& videoPath, float startTime, float endTime, 
                               float interval, const std::string& outputDir);

    /**
     * Checks if initialized
     * @return True if initialized
     */
    bool isInitialized() const;

private:
    /**
     * Detects faces in a frame
     * @param frame Input video frame
     * @return Vector of detected faces
     */
    std::vector<cv::Rect> detectFaces(const cv::Mat& frame);

    bool m_initialized = false;
    cv::CascadeClassifier m_faceClassifier;
};