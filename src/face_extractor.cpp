#include "face_extractor.h"

#include <iostream>
#include <string>
#include <vector>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/imgproc.hpp>

namespace fs = std::filesystem;

FaceExtractor::FaceExtractor() {
    std::string cascadePath = "data/haarcascade_frontalface_default.xml";
    if (!m_faceClassifier.load(cascadePath)) {
        std::cerr << "Error loading face cascade classifier: " << cascadePath << std::endl;
        std::cerr << "Please ensure the file exists at the specified path relative to the executable or project root." << std::endl;
        m_initialized = false;
        return;
    }
    m_initialized = true;
}

bool FaceExtractor::isInitialized() const {
    return m_initialized;
}

bool FaceExtractor::extractFaces(const std::string& videoPath, float timeInSeconds, const std::string& outputDir) {
    if (!m_initialized) {
        std::cerr << "Face extractor not properly initialized. Face detection model might be missing." << std::endl;
        return false;
    }

    try {
        if (!fs::exists(outputDir)) {
            fs::create_directories(outputDir);
        }
    } catch (const fs::filesystem_error& e) {
        std::cerr << "Error creating output directory '" << outputDir << "': " << e.what() << std::endl;
        return false;
    }

    cv::VideoCapture video(videoPath);
    if (!video.isOpened()) {
        std::cerr << "Error: Could not open video file: " << videoPath << std::endl;
        return false;
    }

    double fps = video.get(cv::CAP_PROP_FPS);
    double totalFrames = video.get(cv::CAP_PROP_FRAME_COUNT);

    if (fps == 0) {
        std::cerr << "Error: Could not get FPS from video (or FPS is 0). Cannot process timestamp." << std::endl;
        video.release();
        return false;
    }
    double current_duration = totalFrames / fps;

    if (timeInSeconds < 0 || timeInSeconds > current_duration) {
        std::cerr << "Error: Timestamp " << timeInSeconds << "s is outside video duration of "
                    << current_duration << "s (Total frames: " << totalFrames << ", FPS: " << fps << ")" << std::endl;
        video.release();
        return false;
    }

    video.set(cv::CAP_PROP_POS_FRAMES, static_cast<int>(timeInSeconds * fps));

    cv::Mat frame;
    if (!video.read(frame) || frame.empty()) {
        std::cerr << "Error: Could not read frame or frame is empty at timestamp " << timeInSeconds << "s from video " << videoPath << std::endl;
        video.release();
        return false;
    }
    video.release();

    std::vector<cv::Rect> faces = detectFaces(frame);
    if (faces.empty()) {
        std::cout << "No faces detected at timestamp " << timeInSeconds << "s in video " << videoPath << std::endl;
        return true;
    }

    for (size_t i = 0; i < faces.size(); i++) {
        cv::Rect faceRect = faces[i];
        if (faceRect.x < 0) faceRect.x = 0;
        if (faceRect.y < 0) faceRect.y = 0;
        if (faceRect.x + faceRect.width > frame.cols) faceRect.width = frame.cols - faceRect.x;
        if (faceRect.y + faceRect.height > frame.rows) faceRect.height = frame.rows - faceRect.y;

        if (faceRect.width <= 0 || faceRect.height <= 0) continue;

        cv::Mat faceROI = frame(faceRect);

        fs::path outputDirPath(outputDir);
        std::string filename = "face_" +
                             std::to_string(static_cast<int>(timeInSeconds)) + "s_" +
                             std::to_string(i) + ".jpg";
        fs::path outputPath = outputDirPath / filename;

        if (!cv::imwrite(outputPath.string(), faceROI)) {
            std::cerr << "Error saving face " << i+1 << " to " << outputPath.string() << std::endl;
        } else {
            std::cout << "Saved face " << i+1 << " to " << outputPath.string() << std::endl;
        }
    }

    std::cout << "Extracted " << faces.size() << " faces at timestamp " << timeInSeconds << "s from video " << videoPath << std::endl;
    return true;
}
    
bool FaceExtractor::extractFacesFromRange(const std::string& videoPath, float startTime, float endTime,
                                          float interval, const std::string& outputDir) {
    if (!m_initialized) {
        std::cerr << "Face extractor not properly initialized. Face detection model might be missing." << std::endl;
        return false;
    }

    if (interval <= 0) {
        std::cerr << "Error: Interval for face extraction must be positive." << std::endl;
        return false;
    }

    cv::VideoCapture video_check(videoPath);
    if (!video_check.isOpened()) {
        std::cerr << "Error: Could not open video file for range check: " << videoPath << std::endl;
        return false;
    }
    double fps = video_check.get(cv::CAP_PROP_FPS);
    double totalFrames = video_check.get(cv::CAP_PROP_FRAME_COUNT);
    video_check.release();

    if (fps == 0) {
        std::cerr << "Error: Could not get FPS from video (or FPS is 0). Cannot process range." << std::endl;
        return false;
    }
    double duration = totalFrames / fps;

    if (startTime < 0) startTime = 0;
    if (endTime > duration) endTime = duration;
    if (startTime >= endTime && !(startTime == endTime && startTime == 0 && duration == 0)) {
         if (startTime == endTime && duration > 0) {
         } else {
            std::cerr << "Error: Start time (" << startTime << ") must generally be less than end time (" << endTime << "). Video duration: " << duration << std::endl;
            return false;
         }
    }

    bool all_successful = true;
    for (float time = startTime; time <= endTime; time += interval) {
        std::cout << "--- Processing timestamp: " << time << "s (Range: " << startTime << "-" << endTime << ", Interval: " << interval << ") ---" << std::endl;
        if (!extractFaces(videoPath, time, outputDir)) {
            all_successful = false;
            std::cerr << "Failed to extract faces at timestamp " << time << "s. Continuing with next interval." << std::endl;
        }
        if (time == endTime) break;
        if (time + interval > endTime && time < endTime) {
             if (!extractFaces(videoPath, endTime, outputDir)) {
                all_successful = false;
                std::cerr << "Failed to extract faces at end timestamp " << endTime << "s. Continuing with next interval." << std::endl;
             }
             break;
        }
    }

    std::cout << "--- Face extraction from range completed for video " << videoPath << " --- " << std::endl;
    return all_successful;
}

std::vector<cv::Rect> FaceExtractor::detectFaces(const cv::Mat& frame) {
    std::vector<cv::Rect> faces_detected;
    if (frame.empty()) {
        std::cerr << "Cannot detect faces in an empty frame." << std::endl;
        return faces_detected;
    }

    cv::Mat grayFrame;
    cv::cvtColor(frame, grayFrame, cv::COLOR_BGR2GRAY);
    cv::equalizeHist(grayFrame, grayFrame);

    m_faceClassifier.detectMultiScale(
        grayFrame, faces_detected,
        1.1,
        3,
        0,
        cv::Size(30, 30)
    );

    return faces_detected;
}

void printUsage(const char* programName) {
    std::cout << "Face Extractor - Extracts faces from a video at specific timestamps or ranges." << std::endl;
    std::cout << "Usage:" << std::endl;
    std::cout << "  " << programName << " <video_path> <timestamp_seconds> <output_directory>" << std::endl;
    std::cout << "  " << programName << " --range <video_path> <start_time_seconds> <end_time_seconds> <interval_seconds> <output_directory>" << std::endl;
    std::cout << "Examples:" << std::endl;
    std::cout << "  " << programName << " video.mp4 10.5 faces_output/" << std::endl;
    std::cout << "  " << programName << " --range video.mp4 5.0 15.0 1.0 faces_output/" << std::endl;
    std::cout << "Note: Ensure the output directory exists or can be created." << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        printUsage(argv[0]);
        return 1;
    }

    FaceExtractor extractor; // Initialize here to load models once
    if (!extractor.isInitialized()) {
        std::cerr << "Failed to initialize FaceExtractor. Exiting." << std::endl;
        return 1;
    }

    try {
        if (std::string(argv[1]) == "--range") {
            if (argc != 7) {
                std::cerr << "Error: Incorrect number of arguments for --range mode." << std::endl;
                printUsage(argv[0]);
                return 1;
            }
            std::string videoPath = argv[2];
            float startTime = std::stof(argv[3]);
            float endTime = std::stof(argv[4]);
            float interval = std::stof(argv[5]);
            std::string outputDir = argv[6];
            
            if (!extractor.extractFacesFromRange(videoPath, startTime, endTime, interval, outputDir)) {
                std::cerr << "Face extraction from range encountered errors." << std::endl;
                return 1;
            }
        } else {
            if (argc != 4) {
                std::cerr << "Error: Incorrect number of arguments for single timestamp mode." << std::endl;
                printUsage(argv[0]);
                return 1;
            }
            std::string videoPath = argv[1];
            float timestamp = std::stof(argv[2]);
            std::string outputDir = argv[3];
            
            if (!extractor.extractFaces(videoPath, timestamp, outputDir)) {
                std::cerr << "Face extraction at timestamp failed." << std::endl;
                return 1;
            }
        }
    } catch (const std::invalid_argument& ia) {
        std::cerr << "Error: Invalid numeric argument provided: " << ia.what() << std::endl;
        printUsage(argv[0]);
        return 1;
    } catch (const std::out_of_range& oor) {
        std::cerr << "Error: Numeric argument out of range: " << oor.what() << std::endl;
        printUsage(argv[0]);
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "An unexpected error occurred: " << e.what() << std::endl;
        return 1;
    }

    std::cout << "Face extraction process finished." << std::endl;
    return 0;
}