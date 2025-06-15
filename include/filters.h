#pragma once

#include <vector>
#include <complex>
#include <memory>

/**
 * Audio band-pass filter
 */
class BandPassFilter {
public:
    /**
     * Constructor
     * @param sampleRate Audio sample rate in Hz
     * @param lowCutoff Lower cutoff frequency in Hz
     * @param highCutoff Higher cutoff frequency in Hz
     */
    BandPassFilter(int sampleRate, float lowCutoff, float highCutoff);

    /**
     * Applies the filter
     * @param input Input audio
     * @return Filtered audio
     */
    std::vector<float> apply(const std::vector<float>& input);

private:
    int m_sampleRate;
    float m_lowCutoff;
    float m_highCutoff;
    std::vector<float> m_coefficients;
    
    void calculateCoefficients();
};

/**
 * Noise reduction using spectral subtraction
 */
class SpectralSubtraction {
public:
    /**
     * Constructor
     * @param fftSize FFT size to use for processing
     * @param hopSize Hop size between consecutive frames
     * @param reductionFactor Noise reduction factor (0-1)
     */
    SpectralSubtraction(int fftSize, int hopSize, float reductionFactor);

    /**
     * Processes audio to remove noise
     * @param input Input audio
     * @param noiseProfile Optional noise profile
     * @return Processed audio
     */
    std::vector<float> process(const std::vector<float>& input, const std::vector<float>* noiseProfile = nullptr);

    /**
     * Estimates noise profile
     * @param input Input audio
     * @param durationSec Duration for estimation
     * @return Noise profile
     */
    std::vector<float> estimateNoiseProfile(const std::vector<float>& input, float durationSec = 0.5);

private:
    int m_fftSize;
    int m_hopSize;
    float m_reductionFactor;
    
    void fft_complex_inplace(std::vector<std::complex<float>>& buffer);
    std::vector<std::complex<float>> performFFT(const std::vector<float>& input, int start, int size);
    std::vector<float> performIFFT(const std::vector<std::complex<float>>& spectrum);
    std::vector<float> getWindowFunction(int size);
};

/**
 * Combines audio filters and processing
 */
class AudioProcessor {
public:
    /**
     * Constructor
     * @param sampleRate Audio sample rate in Hz
     * @param lowCutoff Lower cutoff frequency in Hz
     * @param highCutoff Higher cutoff frequency in Hz
     * @param noiseReduction Noise reduction factor (0-1)
     */
    AudioProcessor(int sampleRate, float lowCutoff, float highCutoff, float noiseReduction);

    /**
     * Processes audio data
     * @param input Input audio
     * @return Processed audio
     */
    std::vector<float> process(const std::vector<float>& input);

private:
    std::unique_ptr<BandPassFilter> m_bandPassFilter;
    std::unique_ptr<SpectralSubtraction> m_spectralSubtraction;
    int m_sampleRate;
};