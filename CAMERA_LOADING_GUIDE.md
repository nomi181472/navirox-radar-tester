# Camera Video Loading - User Guide

## Overview
All cameras now start in **OFFLINE** mode. You must manually load video sources before detection can begin.

## How to Use

### Loading a Video:
1. Each camera cell has a URL input field pre-populated with a default video
2. You can:
   - **Use the default video**: Just click the "Load" button
   - **Enter a custom URL**: Replace the text with your video URL (http:// or https://)
   - **Enter a local file path**: Replace with a local file path (e.g., C:\Videos\my_video.mp4)
3. Click the **"Load"** button to register the video
4. Toggle the camera **ON** (switch to green) to start playback and detection

### Camera States:
- **Red dot + OFFLINE label**: Camera is disabled, no detection running
- **Green dot + Video playing**: Camera is active, detection is running
- **FPS counter**: Shows real-time frame rate when active

### Detection Flow:
```
User loads video → Video registered with inference service → 
Toggle camera ON → Frames sent to YOLO model → 
Detections displayed on tactical map
```

## Why Detection Wasn't Working:

**Before:**
- Cameras auto-started with default videos
- Detection ran immediately on app launch

**Now:**
- Cameras start offline (user control)
- No video = no frames = no detections
- **Solution**: Load a video URL and toggle the camera ON

## Default Video URLs:
- **CAM 1 (FWD)**: Cash counter scene
- **CAM 2 (AFT)**: Car traffic
- **CAM 3 (PORT)**: Office scene
- **CAM 4 (STBD)**: Crowd scene

All default videos are hosted on AWS S3 and will stream automatically.

## Troubleshooting:

**No detections appearing?**
1. Check if camera is toggled ON (green)
2. Verify video loaded successfully (check console for ✅ message)
3. Ensure FPS counter shows activity
4. Check if video URL is accessible

**Video won't load?**
- For URLs: Check internet connection
- For local files: Verify file path is correct and file exists
- Check console for error messages (❌)
