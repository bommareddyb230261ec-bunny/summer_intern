# What I Did

I created a person re-identification preprocessing pipeline using three Jupyter notebooks.

In `video_to_frames.ipynb`, I loaded the source WhatsApp video with OpenCV, checked that the file exists, created a `saved_frames` folder, and extracted one frame every 0.5 seconds. I saved the frames with frame numbers and timestamps, then listed the extracted frames.

In `person_detection.ipynb`, I loaded the YOLO11n model and processed all images from `saved_frames`. I detected only the `person` class, drew green bounding boxes with confidence scores, and saved the annotated output frames into `detected_frames` for checking the detection results.

In `person_cropping.ipynb`, I created the final cropping step for the pipeline. I installed/imported the required libraries such as OpenCV, PIL, NumPy, tqdm, pathlib, and Ultralytics YOLO. I set the input folder as `saved_frames`, the output folder as `cropped_persons`, and loaded the local `yolo11n.pt` model.

Then I processed every extracted frame one by one. For each frame, I ran YOLO detection with a confidence threshold of `0.2`, selected only the `person` class, checked the bounding box coordinates, clipped them inside the image size, skipped invalid or very small boxes, and cropped the detected person area from the original frame.

Finally, I saved each cropped person image into the `cropped_persons` folder using names like `frame_<frame_number>_person_<person_number>.jpg`. This notebook also counted how many cropped person images were saved, so I could confirm the final output.

Overall, I converted a video into frames, detected people in those frames, and generated cropped person images that can be used for person re-identification or dataset preparation.
