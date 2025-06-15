#!/bin/bash

# Exit on error
set -e

PROJECT_ROOT=$(pwd)
SRC_DIR="$PROJECT_ROOT/src"
BUILD_DIR="$PROJECT_ROOT/build_bash" 

CXX=g++
CXX_STANDARD="-std=c++17"

# Source files for video_cleaner
APP_SOURCES="$SRC_DIR/main.cpp \
             $SRC_DIR/filters.cpp \
             $SRC_DIR/process.cpp \
             $SRC_DIR/video_denoise.cpp"

# Source file for face_extractor
FACE_EXTRACTOR_SRC="$SRC_DIR/face_extractor.cpp"

# Output executable names
APP_EXECUTABLE="$BUILD_DIR/video_cleaner"
FACE_EXTRACTOR_EXECUTABLE="$BUILD_DIR/face_extractor"

# Create build directory
mkdir -p "$BUILD_DIR"
echo "Build directory: $BUILD_DIR"

# Get compiler and linker flags using pkg-config
echo "Fetching OpenCV flags..."
OPENCV_CFLAGS=$(pkg-config --cflags opencv4)
OPENCV_LIBS=$(pkg-config --libs opencv4)
if [ -z "$OPENCV_CFLAGS" ] || [ -z "$OPENCV_LIBS" ]; then
    echo "Warning: pkg-config failed for opencv4, trying 'opencv'..."
    OPENCV_CFLAGS=$(pkg-config --cflags opencv)
    OPENCV_LIBS=$(pkg-config --libs opencv)
    if [ -z "$OPENCV_CFLAGS" ] || [ -z "$OPENCV_LIBS" ]; then
        echo "Error: Could not get OpenCV flags using pkg-config. Please ensure OpenCV is installed and pkg-config can find it."
        exit 1
    fi
fi
echo "OpenCV CFLAGS: $OPENCV_CFLAGS"
echo "OpenCV LIBS: $OPENCV_LIBS"

echo "Fetching FFmpeg flags..."
FFMPEG_CFLAGS=$(pkg-config --cflags libavcodec libavformat libavutil libswresample)
FFMPEG_LIBS=$(pkg-config --libs libavcodec libavformat libavutil libswresample)
if [ -z "$FFMPEG_LIBS" ]; then
    echo "Error: Could not get FFmpeg linker flags (FFMPEG_LIBS) using pkg-config. Please ensure FFmpeg development libraries are installed."
    exit 1
fi
echo "FFmpeg CFLAGS: $FFMPEG_CFLAGS"
echo "FFmpeg LIBS: $FFMPEG_LIBS"

# Common include paths (src directory for project headers)
# INCLUDE_PATHS="-I$SRC_DIR" # Old way
INCLUDE_PATHS="-Iinclude"

# --- Build video_cleaner ---
echo "Building video_cleaner..."
APP_OBJECTS=""
for src_file in $APP_SOURCES; do
    base_name=$(basename "$src_file" .cpp)
    obj_file="$BUILD_DIR/${base_name}.o"
    echo "Compiling $src_file -> $obj_file"
    $CXX $CXX_STANDARD $INCLUDE_PATHS $OPENCV_CFLAGS $FFMPEG_CFLAGS -c "$src_file" -o "$obj_file"
    APP_OBJECTS="$APP_OBJECTS $obj_file"
done

echo "Linking $APP_EXECUTABLE..."
$CXX $APP_OBJECTS $OPENCV_LIBS $FFMPEG_LIBS -o "$APP_EXECUTABLE"
echo "video_cleaner built successfully: $APP_EXECUTABLE"

# --- Build face_extractor ---
echo "Building face_extractor..."
echo "Compiling $FACE_EXTRACTOR_SRC -> $BUILD_DIR/face_extractor.o"
$CXX $CXX_STANDARD $INCLUDE_PATHS $OPENCV_CFLAGS -c "$FACE_EXTRACTOR_SRC" -o "$BUILD_DIR/face_extractor.o"

echo "Linking $FACE_EXTRACTOR_EXECUTABLE..."
$CXX "$BUILD_DIR/face_extractor.o" $OPENCV_LIBS -o "$FACE_EXTRACTOR_EXECUTABLE"
echo "face_extractor built successfully: $FACE_EXTRACTOR_EXECUTABLE"

echo "Build complete!" 