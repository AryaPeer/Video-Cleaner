#include "filters.h"

#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <numeric>
#include <cassert>

// Constants
const double PI = 3.14159265358979323846;

// BandPassFilter implementation
BandPassFilter::BandPassFilter(int sampleRate, float lowCutoff, float highCutoff)
    : m_sampleRate(sampleRate), m_lowCutoff(lowCutoff), m_highCutoff(highCutoff) {
    
    // Validate parameters
    if (sampleRate <= 0) {
        throw std::invalid_argument("Sample rate must be positive");
    }
    
    if (lowCutoff < 0 || highCutoff < 0 || lowCutoff >= highCutoff) {
        throw std::invalid_argument("Invalid cutoff frequencies");
    }
    
    if (highCutoff >= sampleRate / 2) {
        throw std::invalid_argument("High cutoff must be less than Nyquist frequency");
    }
    
    calculateCoefficients();
}

void BandPassFilter::calculateCoefficients() {
    // Implement a simple FIR bandpass filter design using windowed-sinc
    int filterOrder = 64;  // Filter order
    m_coefficients.resize(filterOrder + 1);
    
    // Normalized frequencies
    float normalizedLow = 2.0f * m_lowCutoff / m_sampleRate;
    float normalizedHigh = 2.0f * m_highCutoff / m_sampleRate;
    
    // Compute filter coefficients
    for (int i = 0; i <= filterOrder; i++) {
        // Handle the special case at center
        if (i == filterOrder / 2) {
            m_coefficients[i] = normalizedHigh - normalizedLow;
        } else {
            float n = i - filterOrder / 2;
            // Bandpass filter is highpass - lowpass
            m_coefficients[i] = (sin(PI * normalizedHigh * n) - sin(PI * normalizedLow * n)) / (PI * n);
            
            // Apply Hamming window
            float windowFactor = 0.54 - 0.46 * cos(2 * PI * i / filterOrder);
            m_coefficients[i] *= windowFactor;
        }
    }
    
    // Normalize the coefficients
    float sum = std::accumulate(m_coefficients.begin(), m_coefficients.end(), 0.0f);
    if (sum != 0) {
        for (auto& coef : m_coefficients) {
            coef /= sum;
        }
    }
}

std::vector<float> BandPassFilter::apply(const std::vector<float>& input) {
    const int filterLength = static_cast<int>(m_coefficients.size());
    const int inputLength = static_cast<int>(input.size());
    std::vector<float> output(inputLength);
    
    // Apply FIR filter
    for (int i = 0; i < inputLength; i++) {
        float sum = 0.0f;
        for (int j = 0; j < filterLength; j++) {
            int inputIdx = i - j;
            
            // Handle edge cases
            if (inputIdx >= 0 && inputIdx < inputLength) {
                sum += input[inputIdx] * m_coefficients[j];
            }
        }
        output[i] = sum;
    }
    
    return output;
}

// SpectralSubtraction implementation
SpectralSubtraction::SpectralSubtraction(int fftSize, int hopSize, float reductionFactor)
    : m_fftSize(fftSize), m_hopSize(hopSize), m_reductionFactor(reductionFactor) {
    
    // Validate parameters
    if (fftSize <= 0 || (fftSize & (fftSize - 1)) != 0) {
        throw std::invalid_argument("FFT size must be a positive power of 2");
    }
    
    if (hopSize <= 0 || hopSize > fftSize) {
        throw std::invalid_argument("Hop size must be positive and not greater than FFT size");
    }
    
    if (reductionFactor < 0.0f || reductionFactor > 1.0f) {
        throw std::invalid_argument("Reduction factor must be between 0 and 1");
    }
}

std::vector<float> SpectralSubtraction::getWindowFunction(int size) {
    // Hann window function
    std::vector<float> window(size);
    for (int i = 0; i < size; i++) {
        window[i] = 0.5f * (1.0f - cos(2.0f * PI * i / (size - 1)));
    }
    return window;
}

void SpectralSubtraction::fft_complex_inplace(std::vector<std::complex<float>>& buffer) {
    int n = buffer.size();
    if (n == 0 || (n & (n - 1)) != 0) { // Must be power of 2
        throw std::runtime_error("FFT size must be a power of 2 for fft_complex_inplace");
    }
    int log2n = 0;
    while ((1 << log2n) < n) {
        log2n++;
    }

    // Bit-reversal permutation
    for (int i = 0; i < n; i++) {
        int j = 0;
        for (int k = 0; k < log2n; k++) {
            j = (j << 1) | ((i >> k) & 1);
        }
        if (j < i) {
            std::swap(buffer[i], buffer[j]);
        }
    }

    // Compute FFT
    for (int s = 1; s <= log2n; s++) {
        int m = 1 << s;
        int m2 = m >> 1;
        std::complex<float> w(1, 0);
        std::complex<float> wm(cos(-2 * PI / m), sin(-2 * PI / m)); // For FFT

        for (int j = 0; j < m2; j++) {
            for (int k = j; k < n; k += m) {
                std::complex<float> t = w * buffer[k + m2];
                std::complex<float> u = buffer[k];
                buffer[k] = u + t;
                buffer[k + m2] = u - t;
            }
            w *= wm;
        }
    }
}

std::vector<std::complex<float>> SpectralSubtraction::performFFT(const std::vector<float>& input, int start, int size) {
    // FFT implementation
    
    std::vector<std::complex<float>> buffer(size);
    
    // Copy input data to buffer with windowing
    auto window = getWindowFunction(size);
    for (int i = 0; i < size; i++) {
        int inputIdx = start + i;
        if (inputIdx < static_cast<int>(input.size())) {
            buffer[i] = std::complex<float>(input[inputIdx] * window[i], 0);
        } else {
            buffer[i] = std::complex<float>(0, 0);
        }
    }
    
    fft_complex_inplace(buffer);
    
    return buffer;
}

std::vector<float> SpectralSubtraction::performIFFT(const std::vector<std::complex<float>>& spectrum) {
    int size = static_cast<int>(spectrum.size());
    if (size == 0) return {};

    std::vector<std::complex<float>> buffer = spectrum;
    for (auto& val : buffer) {
        val = std::conj(val);
    }
    
    fft_complex_inplace(buffer); // Perform FFT on conjugated spectrum
    
    std::vector<float> result(size);
    for (int i = 0; i < size; i++) {
        result[i] = std::conj(buffer[i]).real() / size; // Conjugate again and scale
    }
    
    return result;
}

std::vector<float> SpectralSubtraction::estimateNoiseProfile(const std::vector<float>& input, float durationSec) {
    // Determine number of samples for noise estimation
    int samplesForEstimation = static_cast<int>(44100 * durationSec);  // Assume 44.1kHz sample rate
    samplesForEstimation = std::min(samplesForEstimation, static_cast<int>(input.size()));
    
    // Calculate average spectrum from the first few frames
    std::vector<float> noiseProfile(m_fftSize / 2 + 1, 0.0f);
    int numFrames = 0;
    
    for (int start = 0; start < samplesForEstimation - m_fftSize; start += m_hopSize) {
        auto spectrum = performFFT(input, start, m_fftSize);
        
        // Calculate magnitude spectrum
        for (int i = 0; i <= m_fftSize / 2; i++) {
            float magnitude = std::abs(spectrum[i]);
            noiseProfile[i] += magnitude * magnitude;  // Power spectrum
        }
        
        numFrames++;
    }
    
    // Average the spectra
    if (numFrames > 0) {
        for (auto& val : noiseProfile) {
            val /= numFrames;
        }
    }
    
    return noiseProfile;
}

std::vector<float> SpectralSubtraction::process(const std::vector<float>& input, const std::vector<float>* noiseProfile) {
    // Estimate noise if not provided
    std::vector<float> noise;
    if (noiseProfile == nullptr) {
        noise = estimateNoiseProfile(input);
    } else {
        noise = *noiseProfile;
    }
    
    // Ensure noise profile is the right size
    assert(noise.size() == static_cast<size_t>(m_fftSize / 2 + 1));
    
    // Output buffer
    std::vector<float> output(input.size(), 0.0f);
    
    // Process audio in overlapping frames
    for (size_t start = 0; start + m_fftSize <= input.size(); start += m_hopSize) {
        // Perform FFT
        auto spectrum = performFFT(input, static_cast<int>(start), m_fftSize);
        
        // Apply spectral subtraction
        for (int i = 0; i <= m_fftSize / 2; i++) {
            // Calculate magnitude and phase
            float magnitude = std::abs(spectrum[i]);
            float phase = std::arg(spectrum[i]);
            
            // Subtract noise
            float power = magnitude * magnitude;
            float noisePower = noise[i] * m_reductionFactor;
            float resultPower = std::max(power - noisePower, 0.01f * power);  // Spectral floor
            float resultMagnitude = std::sqrt(resultPower);
            
            // Reconstruct complex spectrum
            spectrum[i] = std::polar(resultMagnitude, phase);
            
            // Mirror for conjugate symmetry
            if (i > 0 && i < m_fftSize / 2) {
                spectrum[m_fftSize - i] = std::conj(spectrum[i]);
            }
        }
        
        // Perform IFFT
        auto frame = performIFFT(spectrum);
        
        // Apply window and overlap-add
        auto window = getWindowFunction(m_fftSize);
        for (int i = 0; i < m_fftSize; i++) {
            if (start + i < output.size()) {
                output[start + i] += frame[i] * window[i];
            }
        }
    }
    
    return output;
}

// AudioProcessor implementation
AudioProcessor::AudioProcessor(int sampleRate, float lowCutoff, float highCutoff, float noiseReduction) 
    : m_sampleRate(sampleRate) {
    
    // Create filters
    m_bandPassFilter = std::make_unique<BandPassFilter>(sampleRate, lowCutoff, highCutoff);
    
    // Create spectral subtraction
    int fftSize = 2048;
    int hopSize = fftSize / 4;  // 75% overlap
    m_spectralSubtraction = std::make_unique<SpectralSubtraction>(fftSize, hopSize, noiseReduction);
}

std::vector<float> AudioProcessor::process(const std::vector<float>& input) {
    // First apply band pass filter
    auto filtered = m_bandPassFilter->apply(input);
    
    // Then apply spectral subtraction
    auto result = m_spectralSubtraction->process(filtered);
    
    return result;
}