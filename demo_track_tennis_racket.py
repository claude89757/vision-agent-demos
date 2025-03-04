import os
import numpy as np
from vision_agent.tools import *
from vision_agent.tools.planner_tools import judge_od_results
from typing import *
from pillow_heif import register_heif_opener
register_heif_opener()
import vision_agent as va
from vision_agent.tools import register_tool

import cv2
import numpy as np
import math
from typing import List, Dict, Any, Optional
from vision_agent.tools import extract_frames_and_timestamps, florence2_sam2_video_tracking
from vision_agent.tools import overlay_bounding_boxes, save_video

def track_racket_speed_trajectory(video_uri: str, output_video_path: str) -> str:
    """
    Tracks the tennis racket in a video, draws its center point trajectory,
    color-codes line segments based on speed, overlays the bounding box of 
    the racket, and saves the result as a new video.

    Parameters:
    -----------
    video_uri : str
        The path or URL of the input video.

    Returns:
    --------
    str
        The path to the saved output video.
    """

    # 1) Extract frames from the video (default fps=5).
    frame_data = extract_frames_and_timestamps(video_uri, fps=5)
    frames = [d["frame"] for d in frame_data]
    if not frames:
        raise ValueError("No frames extracted from video.")

    # 2) Detect the racket bounding boxes in each frame using florence2_sam2_video_tracking.
    tracked_bboxes = florence2_sam2_video_tracking("tennis racket", frames)

    # We'll store center points and compute speeds.
    centers = []
    speeds = []
    prev_center = None

    # Because the bounding boxes are normalized, get the frame size for conversion.
    height, width = frames[0].shape[:2]

    # 3) Convert bounding boxes to center points in pixel coordinates.
    for bboxes in tracked_bboxes:
        if bboxes:
            bbox = bboxes[0]["bbox"]  # take the first bounding box
            x_min_norm, y_min_norm, x_max_norm, y_max_norm = bbox

            x_min = int(x_min_norm * width)
            y_min = int(y_min_norm * height)
            x_max = int(x_max_norm * width)
            y_max = int(y_max_norm * height)

            # Center point
            cx = (x_min + x_max) // 2
            cy = (y_min + y_max) // 2
            centers.append((cx, cy))

            # 4) Compute speed: distance between consecutive centers
            if prev_center is not None:
                dx = cx - prev_center[0]
                dy = cy - prev_center[1]
                speed = math.sqrt(dx*dx + dy*dy)
                speeds.append(speed)
            prev_center = (cx, cy)
        else:
            # If no detection on this frame, just append None
            centers.append(None)
            if prev_center is not None:
                speeds.append(0)

    # Thresholds for color-coding
    if len(speeds) > 0:
        max_speed = max(speeds)
        slow_threshold = max_speed / 3
        medium_threshold = (2 * max_speed) / 3
    else:
        slow_threshold = 0
        medium_threshold = 0

    # 5) Create the output frames
    out_frames = []
    for i, frame in enumerate(frames):
        drawn_frame = frame.copy()

        # Overlays the bounding boxes for clarity. We must pass normalized coordinates.
        # Rebuild a list of bboxes for overlay. We only overlay the known bounding box (if any).
        if tracked_bboxes[i]:
            overlay_input = []
            for item in tracked_bboxes[i]:
                overlay_input.append({
                    "score": item.get("score", 1.0),
                    "label": item.get("label", "tennis_racket"),
                    "bbox": item["bbox"]  # normalized
                })
            drawn_frame = overlay_bounding_boxes(drawn_frame, overlay_input)

        # Color-coded lines for the trajectory
        if i > 0 and centers[i] is not None and centers[i - 1] is not None:
            speed = speeds[i - 1]
            if speed < slow_threshold:
                color = (0, 255, 0)      # Green
            elif speed < medium_threshold:
                color = (0, 255, 255)   # Yellow
            else:
                color = (0, 0, 255)     # Red
            cv2.line(
                drawn_frame,
                centers[i - 1],
                centers[i],
                color,
                thickness=3
            )

        # Draw center point
        if centers[i] is not None:
            cv2.circle(
                drawn_frame,
                centers[i],
                7,
                (255, 0, 0),  # Blue circle
                -1
            )

        # Create a speed legend
        cv2.putText(drawn_frame, "Speed:", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(drawn_frame, "Slow", (120, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(drawn_frame, "Medium", (220, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(drawn_frame, "Fast", (350, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        out_frames.append(drawn_frame)

    # 6) Save the resulting frames as a new video.
    output_path = save_video(out_frames, fps=5, output_video_path=output_video_path)

    return output_path


if __name__ == "__main__":
    data = track_racket_speed_trajectory("./videos/Roger_Federer.mp4", "./videos/Roger_Federer_output.mp4")
    print(data)
