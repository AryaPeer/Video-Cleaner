#include "process.h"

#include <iostream>
#include <vector>
#include <memory>
#include <stdexcept>
#include <fstream> // For std::ofstream
#include <string>  // For std::string manipulations for temp files
#include <cstdio>  // For std::remove

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/avutil.h>
#include <libavutil/opt.h>
#include <libavutil/channel_layout.h>
#include <libswresample/swresample.h>
}

#include "video_denoise.h"
#include "filters.h"

VideoProcessor::VideoProcessor(float lowCutoff, float highCutoff, float noiseReduction, float videoDenoiseStrength)
    : m_lowCutoff(lowCutoff), m_highCutoff(highCutoff), m_noiseReduction(noiseReduction),
      m_videoDenoiseStrength(videoDenoiseStrength) {

    m_audioProcessor = nullptr;
    m_videoDenoiser = createVideoDenoiser(videoDenoiseStrength);
}

bool VideoProcessor::processVideo(const std::string& inputPath, const std::string& outputPath) {
    try {
        std::vector<float> audioData;
        int sampleRate = 0;
        int channels = 0;

        if (!extractAudio(inputPath, audioData, sampleRate, channels)) {
            std::cerr << "Failed to extract audio from video" << std::endl;
            return false;
        }

        if (!processAudio(audioData, sampleRate, channels)) {
            std::cerr << "Failed to process audio" << std::endl;
            return false;
        }

        if (!processVideoFrames(inputPath, outputPath, audioData, sampleRate, channels)) {
            std::cerr << "Failed to process video frames" << std::endl;
            return false;
        }

        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error processing video: " << e.what() << std::endl;
        return false;
    }
}

bool VideoProcessor::extractAudio(const std::string& videoPath, std::vector<float>& audioData, int& sampleRate, int& channels) {
    AVFormatContext* formatContext = nullptr;
    AVCodecContext* codecContext = nullptr;
    AVStream* audioStream = nullptr;
    const AVCodec* codec = nullptr;
    SwrContext* swrContext = nullptr;

    if (avformat_open_input(&formatContext, videoPath.c_str(), nullptr, nullptr) != 0) {
        std::cerr << "Could not open input file: " << videoPath << std::endl;
        return false;
    }

    if (avformat_find_stream_info(formatContext, nullptr) < 0) {
        std::cerr << "Could not find stream information" << std::endl;
        avformat_close_input(&formatContext);
        return false;
    }

    int audioStreamIndex = -1;
    for (unsigned int i = 0; i < formatContext->nb_streams; i++) {
        if (formatContext->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_AUDIO) {
            audioStreamIndex = i;
            break;
        }
    }
    
    if (audioStreamIndex == -1) {
        std::cerr << "Could not find audio stream in video file" << std::endl;
        avformat_close_input(&formatContext);
        return false;
    }

    audioStream = formatContext->streams[audioStreamIndex];

    codec = avcodec_find_decoder(audioStream->codecpar->codec_id);
    if (!codec) {
        std::cerr << "Unsupported audio codec" << std::endl;
        avformat_close_input(&formatContext);
        return false;
    }

    codecContext = avcodec_alloc_context3(codec);
    if (!codecContext) {
        std::cerr << "Failed to allocate audio codec context" << std::endl;
        avformat_close_input(&formatContext);
        return false;
    }

    if (avcodec_parameters_to_context(codecContext, audioStream->codecpar) < 0) {
        std::cerr << "Failed to copy audio codec parameters to context" << std::endl;
        avcodec_free_context(&codecContext);
        avformat_close_input(&formatContext);
        return false;
    }

    if (avcodec_open2(codecContext, codec, nullptr) < 0) {
        std::cerr << "Failed to open audio codec" << std::endl;
        avcodec_free_context(&codecContext);
        avformat_close_input(&formatContext);
        return false;
    }

    sampleRate = codecContext->sample_rate;
    channels = codecContext->ch_layout.nb_channels;

    swrContext = swr_alloc();
    if (!swrContext) {
        std::cerr << "Failed to allocate resampler context" << std::endl;
        avcodec_free_context(&codecContext);
        avformat_close_input(&formatContext);
        return false;
    }

    av_opt_set_int(swrContext, "in_channel_count", codecContext->ch_layout.nb_channels, 0);
    av_opt_set_int(swrContext, "out_channel_count", codecContext->ch_layout.nb_channels, 0);
    av_opt_set_chlayout(swrContext, "in_chlayout", &codecContext->ch_layout, 0);
    av_opt_set_chlayout(swrContext, "out_chlayout", &codecContext->ch_layout, 0);
    av_opt_set_int(swrContext, "in_sample_rate", codecContext->sample_rate, 0);
    av_opt_set_int(swrContext, "out_sample_rate", codecContext->sample_rate, 0);

    if (codecContext->sample_fmt == AV_SAMPLE_FMT_NONE) {
        std::cerr << "Error: Input sample format is AV_SAMPLE_FMT_NONE (invalid/unknown). Cannot configure resampler." << std::endl;
        std::cerr << "Codec: " << codec->name << " (ID: " << codec->id << ")" << std::endl;
        std::cerr << "Sample Fmt from codec: " << codecContext->sample_fmt << std::endl;
        swr_free(&swrContext);
        avcodec_free_context(&codecContext);
        avformat_close_input(&formatContext);
        return false;
    }
    std::cout << "Input sample format: " << av_get_sample_fmt_name(codecContext->sample_fmt) 
              << " (ID: " << codecContext->sample_fmt << ")" << std::endl;

    av_opt_set_sample_fmt(swrContext, "in_sample_fmt", codecContext->sample_fmt, 0);
    av_opt_set_sample_fmt(swrContext, "out_sample_fmt", AV_SAMPLE_FMT_FLTP, 0);

    if (swr_init(swrContext) < 0) {
        std::cerr << "Failed to initialize resampler" << std::endl;
        swr_free(&swrContext);
        avcodec_free_context(&codecContext);
        avformat_close_input(&formatContext);
        return false;
    }

    AVPacket packet;
    AVFrame* frame = av_frame_alloc();
    std::vector<std::vector<float>> channelData(channels);

    av_seek_frame(formatContext, audioStreamIndex, 0, AVSEEK_FLAG_BACKWARD);
    avcodec_flush_buffers(codecContext);

    while (av_read_frame(formatContext, &packet) >= 0) {
        if (packet.stream_index == audioStreamIndex) {
            if (avcodec_send_packet(codecContext, &packet) >= 0) {
                while (avcodec_receive_frame(codecContext, frame) >= 0) {
                    uint8_t** output;
                    int out_samples = frame->nb_samples;
                    av_samples_alloc_array_and_samples(&output, nullptr, channels, out_samples, AV_SAMPLE_FMT_FLTP, 0);

                    out_samples = swr_convert(swrContext, output, out_samples,
                                          (const uint8_t**)frame->extended_data, frame->nb_samples);

                    if (out_samples > 0) {
                        for (int ch = 0; ch < channels; ch++) {
                            float* data = (float*)output[ch];
                            channelData[ch].insert(channelData[ch].end(), data, data + out_samples);
                        }
                    }

                    if (output) {
                        av_freep(&output[0]);
                        av_freep(&output);
                    }
                }
            }
        }
        av_packet_unref(&packet);
    }

    size_t totalSamples = 0;
    for (const auto& ch : channelData) {
        totalSamples = std::max(totalSamples, ch.size());
    }

    audioData.resize(totalSamples * channels);
    for (size_t i = 0; i < totalSamples; i++) {
        for (int ch = 0; ch < channels; ch++) {
            if (i < channelData[ch].size()) {
                audioData[i * channels + ch] = channelData[ch][i];
            } else {
                audioData[i * channels + ch] = 0.0f;
            }
        }
    }

    av_frame_free(&frame);
    swr_free(&swrContext);
    avcodec_free_context(&codecContext);
    avformat_close_input(&formatContext);

    return true;
}

bool VideoProcessor::processAudio(std::vector<float>& audioData, int sampleRate, int channels) {
    if (audioData.empty() || sampleRate <= 0 || channels <= 0) {
        std::cerr << "Invalid audio data or parameters" << std::endl;
        return false;
    }

    m_audioProcessor = std::make_unique<AudioProcessor>(
        sampleRate, m_lowCutoff, m_highCutoff, m_noiseReduction);

    if (channels == 1) {
        audioData = m_audioProcessor->process(audioData);
    } else if (channels >= 2) {
        std::vector<std::vector<float>> channelData(channels);
        for (size_t i = 0; i < channelData.size(); i++) {
            channelData[i].resize(audioData.size() / channels);
        }

        for (size_t i = 0; i < audioData.size() / channels; i++) {
            for (int ch = 0; ch < channels; ch++) {
                channelData[ch][i] = audioData[i * channels + ch];
            }
        }

        for (int ch = 0; ch < channels; ch++) {
            channelData[ch] = m_audioProcessor->process(channelData[ch]);
        }

        size_t maxSamples = 0;
        for (const auto& ch : channelData) {
            maxSamples = std::max(maxSamples, ch.size());
        }

        audioData.resize(maxSamples * channels);
        for (size_t i = 0; i < maxSamples; i++) {
            for (int ch = 0; ch < channels; ch++) {
                if (i < channelData[ch].size()) {
                    audioData[i * channels + ch] = channelData[ch][i];
                } else {
                    audioData[i * channels + ch] = 0.0f;
                }
            }
        }
    }

    return true;
}

cv::Mat VideoProcessor::denoiseFrame(const cv::Mat& frame) {
    static int width = 0;
    static int height = 0;

    if (width != frame.cols || height != frame.rows) {
        width = frame.cols;
        height = frame.rows;
        m_videoDenoiser->initialize(width, height);
    }

    return m_videoDenoiser->denoise(frame);
}

void VideoProcessor::applyAdditionalVideoEnhancements(cv::Mat& frame) {
    double alpha = 1.2;
    int beta = 5;
    frame.convertTo(frame, -1, alpha, beta);
}

void writeWavHeader(std::ofstream& file, int sampleRate, int numChannels, int numSamples, int bitsPerSample) {
    file.write("RIFF", 4);
    int32_t chunkSize = 36 + numSamples * numChannels * (bitsPerSample / 8);
    file.write(reinterpret_cast<const char*>(&chunkSize), 4);
    file.write("WAVE", 4);
    file.write("fmt ", 4);
    int32_t subchunk1Size = 16;
    file.write(reinterpret_cast<const char*>(&subchunk1Size), 4);
    int16_t audioFormat = (bitsPerSample == 32 && numChannels > 0) ? 3 : 1;
    file.write(reinterpret_cast<const char*>(&audioFormat), 2);
    int16_t channels = static_cast<int16_t>(numChannels);
    file.write(reinterpret_cast<const char*>(&channels), 2);
    int32_t sr = sampleRate;
    file.write(reinterpret_cast<const char*>(&sr), 4);
    int32_t byteRate = sampleRate * numChannels * (bitsPerSample / 8);
    file.write(reinterpret_cast<const char*>(&byteRate), 4);
    int16_t blockAlign = numChannels * (bitsPerSample / 8);
    file.write(reinterpret_cast<const char*>(&blockAlign), 2);
    int16_t bps = static_cast<int16_t>(bitsPerSample);
    file.write(reinterpret_cast<const char*>(&bps), 2);
    file.write("data", 4);
    int32_t subchunk2Size = numSamples * numChannels * (bitsPerSample / 8);
    file.write(reinterpret_cast<const char*>(&subchunk2Size), 4);
}

bool VideoProcessor::saveProcessedAudioToWav(const std::string& wavPath, const std::vector<float>& audioData,
                                           int audioSampleRate, int audioChannels) {
    if (audioData.empty()) {
        std::cerr << "Audio data is empty, cannot save WAV file." << std::endl;
        return false;
    }

    std::ofstream outFile(wavPath, std::ios::binary);
    if (!outFile.is_open()) {
        std::cerr << "Failed to open temporary WAV file for writing: " << wavPath << std::endl;
        return false;
    }

    int bitsPerSample = 32;
    int numSamples = audioData.size() / audioChannels;

    writeWavHeader(outFile, audioSampleRate, audioChannels, numSamples, bitsPerSample);

    outFile.write(reinterpret_cast<const char*>(audioData.data()), audioData.size() * sizeof(float));

    if (!outFile.good()) {
        std::cerr << "Error writing to WAV file: " << wavPath << std::endl;
        outFile.close();
        return false;
    }

    outFile.close();
    std::cout << "Processed audio saved to temporary WAV file: " << wavPath << std::endl;
    return true;
}

bool VideoProcessor::processVideoFrames(const std::string& inputPath, const std::string& outputPath,
                                      const std::vector<float>& processedAudio, int audioSampleRate, int audioChannels) {
    cv::VideoCapture inputVideo(inputPath);
    if (!inputVideo.isOpened()) {
        std::cerr << "Could not open input video: " << inputPath << std::endl;
        return false;
    }

    int width = static_cast<int>(inputVideo.get(cv::CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(inputVideo.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps = inputVideo.get(cv::CAP_PROP_FPS);
    int totalFrames = static_cast<int>(inputVideo.get(cv::CAP_PROP_FRAME_COUNT));

    std::string finalOutputPath = outputPath;
    std::string tempVideoFile = finalOutputPath + ".tmp_vid.mp4";

    cv::VideoWriter outputVideo;
    int fourcc = cv::VideoWriter::fourcc('a', 'v', 'c', '1');
    outputVideo.open(tempVideoFile, fourcc, fps, cv::Size(width, height), true);

    if (!outputVideo.isOpened()) {
        std::cerr << "Could not create temporary output video file: " << tempVideoFile << std::endl;
        return false;
    }

    cv::Mat frame;
    int frameCount = 0;

    std::string denoiserType = "CPU";
    std::cout << "Using " << denoiserType << " implementation for video denoising" << std::endl;

    while (inputVideo.read(frame)) {
        cv::Mat denoisedFrame = denoiseFrame(frame);
        applyAdditionalVideoEnhancements(denoisedFrame);
        outputVideo.write(denoisedFrame);

        frameCount++;
        if (frameCount % 100 == 0 || frameCount == totalFrames) {
            std::cout << "Processed " << frameCount << "/" << totalFrames << " frames ("
                      << (100.0 * frameCount / totalFrames) << "%)" << std::endl;
        }
    }

    inputVideo.release();
    outputVideo.release();

    std::string tempAudioPath = finalOutputPath + ".tmp_audio.wav";
    if (!saveProcessedAudioToWav(tempAudioPath, processedAudio, audioSampleRate, audioChannels)) {
        std::cerr << "Failed to save processed audio to temporary WAV file. Muxing aborted." << std::endl;
        return false;
    }

    std::string ffmpeg_cmd = "ffmpeg -y -i \"" + tempVideoFile +
                             "\" -i \"" + tempAudioPath +
                             "\" -c:v copy -c:a aac -strict experimental -shortest \"" +
                             finalOutputPath + "\" 2> ffmpeg_mux_log.txt";

    std::cout << "Executing FFmpeg command: " << ffmpeg_cmd << std::endl;
    int ret = system(ffmpeg_cmd.c_str());

    if (ret == 0) {
        std::cout << "Muxing successful. Final output: " << finalOutputPath << std::endl;
    } else {
        std::cerr << "FFmpeg muxing failed. Return code: " << ret << std::endl;
        std::cerr << "Check ffmpeg_mux_log.txt for details." << std::endl;
        return false;
    }

    if (std::remove(tempVideoFile.c_str()) != 0) {
        std::perror(("Error deleting temporary video file: " + tempVideoFile).c_str());
    }
    if (std::remove(tempAudioPath.c_str()) != 0) {
        std::perror(("Error deleting temporary audio file: " + tempAudioPath).c_str());
    }

    return true;
}