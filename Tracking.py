from pathlib import Path
import cv2
import numpy as np

def detect_color_blob(frame, lower_bounds, upper_bounds, prev_pos=None, search_margin=80, draw_color=(0, 255, 0)):
    h_frame, w_frame = frame.shape[:2]
    roi_offset_x, roi_offset_y = 0, 0
    search_frame = frame

    if prev_pos is not None and prev_pos[0] is not None and prev_pos[1] is not None:
        px, py = prev_pos
        x1 = max(0, int(px - search_margin))
        y1 = max(0, int(py - search_margin))
        x2 = min(w_frame, int(px + search_margin))
        y2 = min(h_frame, int(py + search_margin))

        search_frame = frame[y1:y2, x1:x2]
        roi_offset_x, roi_offset_y = x1, y1

    hsv = cv2.cvtColor(search_frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_bounds[0], upper_bounds[0])
    for i in range(1, len(lower_bounds)):
        mask2 = cv2.inRange(hsv, lower_bounds[i], upper_bounds[i])
        mask = cv2.bitwise_or(mask, mask2)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours and prev_pos is not None:
        return detect_color_blob(frame, lower_bounds, upper_bounds, prev_pos=None, search_margin=search_margin, draw_color=draw_color)

    annotated = frame.copy()

    if not contours:
        return None, None, mask, annotated

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 3:
        if prev_pos is not None:
            return detect_color_blob(frame, lower_bounds, upper_bounds, prev_pos=None, search_margin=search_margin, draw_color=draw_color)
        return None, None, mask, annotated

    M = cv2.moments(largest)
    if M["m00"] == 0:
        return None, None, mask, annotated

    cx = int(M["m10"] / M["m00"]) + roi_offset_x
    cy = int(M["m01"] / M["m00"]) + roi_offset_y

    cv2.circle(annotated, (cx, cy), 8, draw_color, -1)
    return cx, cy, mask, annotated


def save_calibration_frames(video_path, output_dir, frame_index=0, green_to_red_m=0.1507, sigma_pos_px=2.0, sigma_L2_m=0.001):
    """
    Slaat 2 kalibratieafbeeldingen op en berekent pixels_per_meter inclusief onzekerheid sigma_ppm.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    success, frame = cap.read()
    cap.release()

    if not success or frame is None:
        raise ValueError(f"Kon frame {frame_index} niet inlezen van {video_path.name}.")

    red_lowers = [np.array([0, 80, 50]), np.array([170, 80, 50])]
    red_uppers = [np.array([10, 255, 255]), np.array([180, 255, 255])]
    green_lowers = [np.array([35, 50, 50])]
    green_uppers = [np.array([85, 255, 255])]

    rx, ry, _, annotated_red = detect_color_blob(frame, red_lowers, red_uppers, draw_color=(255, 0, 0))
    gx, gy, _, annotated_green = detect_color_blob(frame, green_lowers, green_uppers, draw_color=(0, 0, 255))

    cv2.imwrite(str(output_dir / f"{video_path.stem}_calibration_red.png"), annotated_red)
    cv2.imwrite(str(output_dir / f"{video_path.stem}_calibration_green.png"), annotated_green)

    if rx is None or gx is None:
        raise ValueError(f"Kalibratie mislukt op frame {frame_index}: Rood of groen punt niet gevonden.")

    dx = gx - rx
    dy = gy - ry
    D_pixels = np.sqrt(dx**2 + dy**2)
    pixels_per_meter = D_pixels / green_to_red_m

    # Foutvoortplanting voor D_pixels: sigma_D = sqrt(2) * sigma_pos_px
    sigma_D_pixels = np.sqrt(2) * sigma_pos_px

    # Relatieve fout op ppm: (sigma_ppm / ppm)^2 = (sigma_D / D)^2 + (sigma_L2 / L2)^2
    rel_err_sq = (sigma_D_pixels / D_pixels)**2 + (sigma_L2_m / green_to_red_m)**2
    sigma_ppm = pixels_per_meter * np.sqrt(rel_err_sq)

    return pixels_per_meter, sigma_ppm, (rx, ry, gx, gy)