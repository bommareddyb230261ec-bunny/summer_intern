# What I Did

I worked on a simple person detection pipeline using two Jupyter notebooks inside the `person_reidentification` folder.

First, in `video_to_frames.ipynb`, I loaded a WhatsApp video using OpenCV. I checked whether the video file exists, created a `saved_frames` folder, opened the video, read its FPS, and saved one frame every 0.5 seconds. Each saved frame was named with a frame number and timestamp, then I listed all extracted frames at the end.

Next, in `person_detection.ipynb`, I used the saved frames as input for person detection. I installed and imported OpenCV, NumPy, pathlib, and Ultralytics YOLO. I loaded the `yolo11n.pt` model and processed every image from the `saved_frames` folder.

For each frame, I ran YOLO with a confidence threshold of `0.2` so that partial people could also be detected. I filtered detections to only the `person` class, drew green bounding boxes around detected people, added confidence labels, and saved the processed images into a `detected_frames` folder.

Overall, I converted a video into image frames and then applied YOLO-based person detection on those frames to produce output images with detected persons marked.
