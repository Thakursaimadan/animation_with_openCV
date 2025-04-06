import cv2
import numpy as np
import glob
import random
from pathlib import Path
import moviepy.editor as mpe

# Resize and read image
def preprocess_image(image_path, target_size=(640, 480)):
    image = cv2.imread(image_path)
    image = cv2.resize(image, target_size)
    return image

# Apply sepia effect
def apply_sepia(image):
    kernel = np.array([[0.272, 0.534, 0.131],
                       [0.349, 0.686, 0.168],
                       [0.393, 0.769, 0.189]])
    sepia = cv2.transform(image, kernel)
    sepia = np.clip(sepia, 0, 255)
    return sepia.astype(np.uint8)

# Fade in text with alpha blending
def add_text_overlay(image, text, step, total_steps):
    overlay = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    alpha = min(step / (total_steps / 2), 1.0)
    cv2.putText(overlay, text, (50, 440), font, 1.2, (255, 255, 255), 3, cv2.LINE_AA)
    return cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)

# Animated border
def add_frame_border(image, step, total_steps):
    border_color = (0, 255, 255) if (step // 5) % 2 == 0 else (255, 0, 255)
    return cv2.rectangle(image.copy(), (5, 5), (image.shape[1]-5, image.shape[0]-5), border_color, 4)

# Ken Burns effect with zoom + text + border
def ken_burns_effect(image, steps=30, zoom_range=(1.0, 1.2), text=""):
    frames = []
    h, w = image.shape[:2]
    for i in range(steps):
        zf = zoom_range[0] + (zoom_range[1] - zoom_range[0]) * i / steps
        new_w, new_h = int(w * zf), int(h * zf)
        resized = cv2.resize(image, (new_w, new_h))
        x_start = (new_w - w) * i // steps
        y_start = (new_h - h) * i // steps
        crop = resized[y_start:y_start + h, x_start:x_start + w]
        crop = add_frame_border(crop, i, steps)
        if text:
            crop = add_text_overlay(crop, text, i, steps)
        frames.append(crop)
    return frames

# Transition: Fade
def fade_transition(imageA, imageB, steps=30):
    frames = []
    for i in range(steps):
        alpha = i / float(steps)
        blended = cv2.addWeighted(imageA, 1 - alpha, imageB, alpha, 0)
        frames.append(blended)
    return frames

# Transition: Slide
def slide_transition(imageA, imageB, steps=30):
    frames = []
    height, width = imageA.shape[:2]
    for i in range(steps):
        shift = int(width * (i / float(steps)))
        frame = np.zeros_like(imageA)
        if shift < width:
            frame[:, :width-shift] = imageA[:, shift:]
            frame[:, width-shift:] = imageB[:, :shift]
        else:
            frame = imageB.copy()
        frames.append(frame)
    return frames

# Transition: Wipe
def wipe_transition(imageA, imageB, steps=30):
    frames = []
    height, width = imageA.shape[:2]
    for i in range(steps):
        wipe_pos = int(width * (i / float(steps)))
        frame = imageA.copy()
        frame[:, :wipe_pos] = imageB[:, :wipe_pos]
        frames.append(frame)
    return frames

# Transition: Zoom crossfade
def zoom_transition(imageA, imageB, steps=30):
    frames = []
    height, width = imageA.shape[:2]
    for i in range(steps):
        zfA = 1 - (0.2 * i / steps)
        zfB = 0.8 + (0.2 * i / steps)
        resized_A = cv2.resize(imageA, None, fx=zfA, fy=zfA)
        resized_B = cv2.resize(imageB, None, fx=zfB, fy=zfB)
        frame = np.zeros_like(imageA)
        yA, xA = (height - resized_A.shape[0]) // 2, (width - resized_A.shape[1]) // 2
        yB, xB = (height - resized_B.shape[0]) // 2, (width - resized_B.shape[1]) // 2
        frame[yA:yA+resized_A.shape[0], xA:xA+resized_A.shape[1]] = resized_A
        overlay = frame.copy()
        overlay[yB:yB+resized_B.shape[0], xB:xB+resized_B.shape[1]] = resized_B
        alpha = i / steps
        blended = cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0)
        frames.append(blended)
    return frames

# Save video and optionally add audio
def create_video(frames, output_file, fps=30):
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter("temp_no_audio.mp4", fourcc, fps, (width, height))
    for frame in frames:
        out.write(frame)
    out.release()

    if Path("background.mp3").exists():
        video = mpe.VideoFileClip("temp_no_audio.mp4")
        audio = mpe.AudioFileClip("background.mp3").set_duration(video.duration)
        final = video.set_audio(audio)
        final.write_videofile(output_file, codec="libx264")
    else:
        Path("temp_no_audio.mp4").rename(output_file)

# Main controller
def main():
    image_paths = sorted(glob.glob("photos/*.jpeg"))
    if not image_paths:
        print("No images found.")
        return

    captions = ["Welcome", "Our Journey", "Memories", "The End"]
    images = [apply_sepia(preprocess_image(p)) for p in image_paths]
    all_frames = []
    transition_funcs = [fade_transition, slide_transition, wipe_transition, zoom_transition]

    for i in range(len(images)-1):
        caption = captions[i] if i < len(captions) else f"Slide {i+1}"
        all_frames.extend(ken_burns_effect(images[i], steps=30, text=caption))
        transition = random.choice(transition_funcs)
        all_frames.extend(transition(images[i], images[i+1], steps=30))

    all_frames.extend(ken_burns_effect(images[-1], steps=30, text="Thank You"))
    create_video(all_frames, "enhanced_animation.mp4")
    print("Enhanced video saved as enhanced_animation.mp4")

if __name__ == "__main__":
    main()
