# Video Cleaner

A video processing tool that cleans video and removes background noise from audio.

## Features
- Custom band pass filtering for audio
- Spectral subtraction for background noise removal
- Video denoising (CPU based)
- Face extraction from video timestamps

## Requirements
- OpenCV (4.0+)
- FFmpeg libraries (libavcodec, libavformat, libavutil, libswresample)

## Installation

### Install dependencies (Ubuntu/Debian)
```bash
# Install build tools
sudo apt-get update
sudo apt-get install -y build-essential pkg-config

# Install OpenCV
sudo apt-get install -y libopencv-dev

# Install FFmpeg development libraries
sudo apt-get install -y libavcodec-dev libavformat-dev libavutil-dev libswresample-dev
```

### Install dependencies (Manjaro/Arch)
```bash
# Install build tools
sudo pacman -Syu
sudo pacman -S --needed base-devel pkg-config

# Install OpenCV
sudo pacman -S opencv

# Install FFmpeg
sudo pacman -S ffmpeg
```

## Building
```bash
./build.sh
```
This will create a `build_bash` directory containing the `video_cleaner` and `face_extractor` executables.

## Usage

### Video Cleaner
```bash
./video_cleaner input_video.mp4 output_video.mp4
```

#### Options
- `--low-cutoff` (default: 100): Low cutoff frequency for bandpass filter in Hz
- `--high-cutoff` (default: 8000): High cutoff frequency for bandpass filter in Hz
- `--noise-reduction` (default: 0.5): Spectral subtraction noise reduction factor (0-1)
- `--video-denoise-strength` (default: 10): Video denoising strength (0-100)

### Face Extractor
Extract faces from a video at specific timestamps:

```bash
# Extract faces from a single timestamp
./face_extractor video.mp4 10.5 faces/

# Extract faces from a time range with specified interval
./face_extractor --range video.mp4 5.0 15.0 1.0 faces/
```

#### Face Extractor Options
- Single timestamp: `./face_extractor video_path timestamp output_directory`
- Time range: `./face_extractor --range video_path start_time end_time interval output_directory`

## Performance Notes

- For best performance use a smaller `--video-denoise-strength` value
