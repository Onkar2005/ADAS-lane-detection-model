# lane_detection.py
import cv2
import numpy as np
import time
from collections import deque

# ---------- Configurable parameters ----------
WEBCAM_INDEX = "video1.mp4"        # or "video.mp4"
FRAME_WIDTH = 960
FRAME_HEIGHT = 540
GAUSSIAN_KERNEL = (5, 5)
CANNY_LOW = 50
CANNY_HIGH = 150
HOUGH_RHO = 2
HOUGH_THETA = np.pi / 180
HOUGH_THRESHOLD = 50
MIN_LINE_LEN = 40
MAX_LINE_GAP = 150
SMOOTHING_FRAMES = 8      # average lane lines over this many frames
# ---------------------------------------------

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    if len(mask.shape) > 2:
        channel_count = mask.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    return cv2.bitwise_and(img, mask)

def draw_line(img, line, color=(0,255,0), thickness=8):
    if line is None:
        return
    x1, y1, x2, y2 = line
    cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def draw_lines(img, lines, color=(0,255,0), thickness=8):
    if lines is None:
        return
    for line in lines:
        if line is None:
            continue
        draw_line(img, line, color, thickness)

def make_coordinates(img_shape, slope, intercept):
    if slope == 0 or slope is None:
        return None
    y1 = img_shape[0]
    y2 = int(y1 * 0.6)
    # x = (y - b) / m
    try:
        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)
    except Exception:
        return None
    return [x1, y1, x2, y2]

def average_slope_intercept(img, lines):
    left_lines = []
    right_lines = []
    if lines is None:
        return None, None

    for l in lines:
        for x1, y1, x2, y2 in l:
            if x2 == x1:
                continue  # skip vertical
            slope = (y2 - y1) / (x2 - x1 + 1e-6)
            intercept = y1 - slope * x1
            if slope < -0.3:            # empirical filter to reject near-horizontal
                left_lines.append((slope, intercept))
            elif slope > 0.3:
                right_lines.append((slope, intercept))

    left_avg = None
    right_avg = None
    if left_lines:
        left_mean = np.mean(left_lines, axis=0)
        left_avg = make_coordinates(img.shape, left_mean[0], left_mean[1])
    if right_lines:
        right_mean = np.mean(right_lines, axis=0)
        right_avg = make_coordinates(img.shape, right_mean[0], right_mean[1])

    return left_avg, right_avg

def calculate_deviation(frame_width, left_line, right_line):
    # deviation in percentage of half-width: positive => vehicle is right of lane center
    if left_line is None or right_line is None:
        return None
    # bottom x positions
    try:
        left_x = left_line[0]
        right_x = right_line[0]
    except Exception:
        return None
    lane_center = (left_x + right_x) / 2.0
    frame_center = frame_width / 2.0
    deviation_pixels = frame_center - lane_center
    deviation_pct = (deviation_pixels / (frame_width / 2.0)) * 100.0
    return int(deviation_pct)  # rounded percent

def overlay_transparent(base_img, overlay_img, alpha=0.4):
    """Alpha blend overlay_img onto base_img where overlay non-zero."""
    if overlay_img.shape[2] != 3:
        return base_img
    mask = (overlay_img.sum(axis=2) > 0).astype(np.uint8)
    mask3 = np.stack([mask]*3, axis=2)
    blended = base_img.copy().astype(float)
    blended[mask3==1] = (1 - alpha) * blended[mask3==1] + alpha * overlay_img[mask3==1]
    return blended.astype(np.uint8)

def process_frame(frame, smoothing_deque):
    height, width = frame.shape[:2]

    # Preprocess
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, GAUSSIAN_KERNEL, 0)
    edges = cv2.Canny(blur, CANNY_LOW, CANNY_HIGH)

    # ROI polygon (trapezoid)
    bottom_left = (int(width * 0.08), height)
    bottom_right = (int(width * 0.92), height)
    top_right = (int(width * 0.58), int(height * 0.6))
    top_left = (int(width * 0.42), int(height * 0.6))
    roi_vertices = np.array([[bottom_left, bottom_right, top_right, top_left]], dtype=np.int32)

    cropped_edges = region_of_interest(edges, roi_vertices)

    # Hough lines
    lines = cv2.HoughLinesP(cropped_edges,
                            rho=HOUGH_RHO,
                            theta=HOUGH_THETA,
                            threshold=HOUGH_THRESHOLD,
                            minLineLength=MIN_LINE_LEN,
                            maxLineGap=MAX_LINE_GAP)

    left_line, right_line = average_slope_intercept(frame, lines)

    # smoothing
    smoothing_deque.append((left_line, right_line))
    if len(smoothing_deque) > SMOOTHING_FRAMES:
        smoothing_deque.popleft()

    # compute averaged smoothed lines
    left_smoothed = None
    right_smoothed = None
    left_vals = [x[0] for x in smoothing_deque if x[0] is not None]
    right_vals = [x[1] for x in smoothing_deque if x[1] is not None]
    if left_vals:
        left_smoothed = np.mean(left_vals, axis=0).astype(int).tolist()
    if right_vals:
        right_smoothed = np.mean(right_vals, axis=0).astype(int).tolist()

    # draw lanes on transparent overlay
    lane_overlay = np.zeros_like(frame)
    if left_smoothed is not None and right_smoothed is not None:
        # draw boundary lines (green)
        draw_line(lane_overlay, left_smoothed, color=(0,255,0), thickness=8)
        draw_line(lane_overlay, right_smoothed, color=(0,255,0), thickness=8)

        # fill polygon (blue)
        poly_pts = np.array([[ [left_smoothed[0], left_smoothed[1]],
                               [left_smoothed[2], left_smoothed[3]],
                               [right_smoothed[2], right_smoothed[3]],
                               [right_smoothed[0], right_smoothed[1]] ]], dtype=np.int32)
        cv2.fillPoly(lane_overlay, poly_pts, (255, 0, 0))  # blue fill

    # overlay with transparency
    combined = overlay_transparent(frame, lane_overlay, alpha=0.5)

    deviation = calculate_deviation(width, left_smoothed, right_smoothed)
    lanes_detected = (left_smoothed is not None and right_smoothed is not None)
    return combined, deviation, lanes_detected, roi_vertices

def draw_info_panel(img, deviation, fps, brightness, lanes_detected):
    # left info box
    cv2.rectangle(img, (10, 10), (330, 210), (40, 40, 160), 2)
    cv2.putText(img, "<LANE DETECTION>", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230,230,230), 2)

    dev_text = f"Deviation: {deviation:+}%" if deviation is not None else "Deviation: N/A"
    cv2.putText(img, dev_text, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
    cv2.putText(img, f"FPS: {fps:.2f}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
    cv2.putText(img, f"Brightness: {brightness}", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

    if lanes_detected:
        cv2.putText(img, "Good Visibility", (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,200,0), 2)
        cv2.putText(img, "Both Lane Detected", (20, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,200,0), 2)
    else:
        cv2.putText(img, "Lane Lost", (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,0,255), 2)

    # right status box
    h, w = img.shape[:2]
    cv2.rectangle(img, (w - 360, 10), (w - 10, 140), (40, 40, 160), 2)
    cv2.putText(img, "[Lane Keeping Status]", (w - 350, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230,230,230), 2)

    if deviation is None:
        cv2.putText(img, "Searching Lanes...", (w - 350, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,200,200), 2)
        cv2.putText(img, "[Upcoming Road]: Unknown", (w - 350, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220,220,220), 1)
        # vehicle speed very low or not in auto mode because searching lanes and warn system activate... 

    else:
        if abs(deviation) < 8:
            cv2.putText(img, "Good Lane Keeping", (w - 350, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,220,0), 2)
            cv2.putText(img, "[Upcoming Road]: Stay Straight", (w - 350, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220,220,220), 1)
            # Vehicle good in lane cancontinue in straight at avg speed 

        elif deviation > 0:
            cv2.putText(img, "Drifting Right!", (w - 350, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,180,255), 2)
            cv2.putText(img, "[Upcoming Road]: Adjust Left", (w - 350, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220,220,220), 1)
            # vehicle drifting right side slight adjust at left side 

        else:
            cv2.putText(img, "Drifting Left!", (w - 350, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,180,255), 2)
            cv2.putText(img, "[Upcoming Road]: Adjust Right", (w - 350, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220,220,220), 1)
            # vehicle drifting left side slight adjust at right side 

def main():
    cap = cv2.VideoCapture(WEBCAM_INDEX)
    if not cap.isOpened():
        print("ERROR: Could not open video source.")
        return

    smoothing_deque = deque()
    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # resize for consistent processing
        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

        combined, deviation, lanes_detected, roi_vertices = process_frame(frame, smoothing_deque)

        # compute fps and brightness
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time + 1e-6)
        prev_time = curr_time
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        brightness = int(np.mean(hsv[:, :, 2]))

        # draw ROI polygon outline for debugging/visual
        cv2.polylines(combined, roi_vertices, isClosed=True, color=(0,255,255), thickness=2)

        # draw small center marker
        h, w = combined.shape[:2]
        cv2.line(combined, (w//2, h - 20), (w//2, h - 5), (255,255,255), 3)

        # info panels
        draw_info_panel(combined, deviation, fps, brightness, lanes_detected)

        cv2.imshow("Lane Detection - Press 'q' to quit", combined)
    

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
