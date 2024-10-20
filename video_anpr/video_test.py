import os
import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('../weights/best.pt')

# Open the video file
video_dir = "test_samples"

# List all files in the directory
video_files = [f for f in os.listdir(video_dir) if f.endswith(('.mp4', '.avi', '.mov'))]

# If there are video files, pick the first one (or modify to choose specific one)
if video_files:
    video_path = os.path.join(video_dir, video_files[0])  # Adjust the index to select different video if necessary
    print(f"Detected video: {video_path}")
else:
    raise FileNotFoundError("No video files found in the 'test_samples' directory.")

# Output directory for the processed video
video_output = "video_outputs/"

# Create the output directory if it doesn't exist
if not os.path.exists(video_output):
    os.makedirs(video_output)

# Capture video
cap = cv2.VideoCapture(video_path)

# Define output path and name
output_video = os.path.join(video_output, 'output_video.mp4')
print("Saving to:", output_video)

# Set up the VideoWriter with MP4 codec
cap_out = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'mp4v'), cap.get(cv2.CAP_PROP_FPS), (1280, 720))

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Resize the frame to the desired resolution
        resize = cv2.resize(frame, (1280, 720))

        # Run YOLOv8 inference on the frame
        results = model.predict(resize)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Write the annotated frame to the output video
        cap_out.write(annotated_frame)

        # Display the annotated frame
        cv2.imshow("YOLOv8 Inference", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture and writer objects and close the display window
cap.release()
cap_out.release()
cv2.destroyAllWindows()
