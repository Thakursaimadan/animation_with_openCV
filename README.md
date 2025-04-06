# 🎞️ Enhanced Image Transition Animation

This project creates a stunning animated video from a set of images using OpenCV and MoviePy. It features a variety of smooth transitions (fade, slide, wipe, zoom), dynamic text overlays, animated borders, the Ken Burns effect (pan & zoom), and optional background music.

## ✨ Features

- 📷 **Image Preprocessing**: Resize and apply sepia filter
- 🎬 **Ken Burns Effect**: Smooth zoom-in/out motion on static images
- 💬 **Text Overlays**: Captions fade in dynamically on each slide
- 🖼️ **Animated Borders**: Flickering borders that animate per frame
- 🔄 **Transitions**:
  - Fade
  - Slide
  - Wipe
  - Zoom crossfade
- 🎵 **Optional Background Music**: Add `background.mp3` to auto-overlay audio
- 📹 **Video Export**: Saves final result as `enhanced_animation.mp4`

## 🔧 Requirements

Install dependencies using pip:

```bash
pip install opencv-python moviepy numpy
```
🏁 How to Run
Add your .jpeg images to the photos/ directory.

(Optional) Add a background.mp3 file for background music.

Run the script:

```bash
python script.py
```
The final video will be saved as enhanced_animation.mp4.

📝 Captions
The following default captions are used:

python
captions = ["Welcome", "Our Journey", "Memories", "The End"]
You can edit or extend this list inside the script to customize text overlays.

🔄 Transition Types
Each transition between images is randomly selected from:

fade_transition

slide_transition

wipe_transition

zoom_transition

📦 Output
The output video is saved at 30 FPS and sized 640x480.

If background.mp3 is present, it will be synced and embedded.

📌 Notes
Only .jpeg images are processed. You can modify the glob pattern to support other formats.

Ensure images are placed in the photos/ folder.
