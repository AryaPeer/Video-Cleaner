#include "main.h"
#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <cstring>

#include "process.h"
#include "video_denoise.h"

void printUsage(const char* programName) {
    std::cout << "Video Cleaner - Removes background noise and cleans video" << std::endl;
    std::cout << "Usage: " << programName << " [options] input_video output_video" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  --low-cutoff <Hz>           : Low cutoff frequency for bandpass filter (default: 100)" << std::endl;
    std::cout << "  --high-cutoff <Hz>          : High cutoff frequency for bandpass filter (default: 8000)" << std::endl;
    std::cout << "  --noise-reduction <0-1>     : Spectral subtraction noise reduction factor (default: 0.5)" << std::endl;
    std::cout << "  --video-denoise-strength <0-100> : Video denoising strength (default: 10)" << std::endl;
    std::cout << "  --help, -h                  : Display this help message" << std::endl;
}

int main(int argc, char* argv[]) {
    float lowCutoff = 100.0f;
    float highCutoff = 8000.0f;
    float noiseReduction = 0.5f;
    float videoDenoiseStrength = 10.0f;
    std::string inputPath;
    std::string outputPath;

    int argIdx = 1;
    while (argIdx < argc) {
        if (strcmp(argv[argIdx], "--low-cutoff") == 0 && argIdx + 1 < argc) {
            lowCutoff = std::stof(argv[argIdx + 1]);
            argIdx += 2;
        } else if (strcmp(argv[argIdx], "--high-cutoff") == 0 && argIdx + 1 < argc) {
            highCutoff = std::stof(argv[argIdx + 1]);
            argIdx += 2;
        } else if (strcmp(argv[argIdx], "--noise-reduction") == 0 && argIdx + 1 < argc) {
            noiseReduction = std::stof(argv[argIdx + 1]);
            argIdx += 2;
        } else if (strcmp(argv[argIdx], "--video-denoise-strength") == 0 && argIdx + 1 < argc) {
            videoDenoiseStrength = std::stof(argv[argIdx + 1]);
            argIdx += 2;
        } else if (strcmp(argv[argIdx], "--help") == 0 || strcmp(argv[argIdx], "-h") == 0) {
            printUsage(argv[0]);
            return 0;
        } else if (inputPath.empty()) {
            inputPath = argv[argIdx++];
        } else if (outputPath.empty()) {
            outputPath = argv[argIdx++];
        } else {
            std::cerr << "Unexpected argument: " << argv[argIdx] << std::endl;
            printUsage(argv[0]);
            return 1;
        }
    }

    if (inputPath.empty() || outputPath.empty()) {
        std::cerr << "Error: Input and output video paths are required" << std::endl;
        printUsage(argv[0]);
        return 1;
    }

    if (lowCutoff < 0 || highCutoff <= lowCutoff) {
        std::cerr << "Error: Invalid cutoff frequencies" << std::endl;
        return 1;
    }
    
    if (noiseReduction < 0 || noiseReduction > 1) {
        std::cerr << "Error: Noise reduction must be between 0 and 1" << std::endl;
        return 1;
    }
    
    if (videoDenoiseStrength < 0 || videoDenoiseStrength > 100) {
        std::cerr << "Error: Video denoise strength must be between 0 and 100" << std::endl;
        return 1;
    }

    try {
        std::cout << "Processing video with the following parameters:" << std::endl;
        std::cout << "  Low cutoff: " << lowCutoff << " Hz" << std::endl;
        std::cout << "  High cutoff: " << highCutoff << " Hz" << std::endl;
        std::cout << "  Noise reduction: " << noiseReduction << std::endl;
        std::cout << "  Video denoise strength: " << videoDenoiseStrength << std::endl;
        
        VideoProcessor processor(lowCutoff, highCutoff, noiseReduction, videoDenoiseStrength);
        bool success = processor.processVideo(inputPath, outputPath);
        
        if (success) {
            std::cout << "Video processing completed successfully!" << std::endl;
            std::cout << "Output saved to: " << outputPath << std::endl;
            return 0;
        } else {
            std::cerr << "Video processing failed!" << std::endl;
            return 1;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}