from pathlib import Path
import cv2
import numpy as np


def detect_color_blob(
    frame,
    lower_bounds,
    upper_bounds,
    prev_pos=None,
    search_margin=80,
    draw_color=(0, 255, 0),
    min_area=10,
    max_area=1500,
    y_min_ratio=0.30,  # Negeer de bovenste 35% van het scherm (ramen/bomen)
):
  h_frame, w_frame = frame.shape[:2]
  y_min_pixel = int(h_frame * y_min_ratio)

  roi_offset_x, roi_offset_y = 0, 0
  search_frame = frame

  # 1. Gebruik zoekraster als er een vorige positie bekend is
  if (
      prev_pos is not None
      and prev_pos[0] is not None
      and prev_pos[1] is not None
  ):
    px, py = prev_pos
    x1 = max(0, int(px - search_margin))
    y1 = max(y_min_pixel, int(py - search_margin))  # Blijf altijd onder y_min
    x2 = min(w_frame, int(px + search_margin))
    y2 = min(h_frame, int(py + search_margin))

    if y2 > y1 and x2 > x1:
      search_frame = frame[y1:y2, x1:x2]
      roi_offset_x, roi_offset_y = x1, y1
    else:
      search_frame = frame[y_min_pixel:, :]
      roi_offset_x, roi_offset_y = 0, y_min_pixel
  else:
    # Eerste frame: zoek uitsluitend onder de kartongrens
    search_frame = frame[y_min_pixel:, :]
    roi_offset_x, roi_offset_y = 0, y_min_pixel

  hsv = cv2.cvtColor(search_frame, cv2.COLOR_BGR2HSV)
  mask = cv2.inRange(hsv, lower_bounds[0], upper_bounds[0])
  for i in range(1, len(lower_bounds)):
    mask2 = cv2.inRange(hsv, lower_bounds[i], upper_bounds[i])
    mask = cv2.bitwise_or(mask, mask2)

  kernel = np.ones((5, 5), np.uint8)
  mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
  mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

  contours, _ = cv2.findContours(
      mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
  )

  valid_contours = [
      c for c in contours if min_area <= cv2.contourArea(c) <= max_area
  ]

  # Fallback: als we niets vonden binnen de zoekmarge, zoek op het hele onderstel (onder y_min)
  if not valid_contours and prev_pos is not None:
    return detect_color_blob(
        frame,
        lower_bounds,
        upper_bounds,
        prev_pos=None,
        search_margin=search_margin,
        draw_color=draw_color,
        min_area=min_area,
        max_area=max_area,
        y_min_ratio=y_min_ratio,
    )

  annotated = frame.copy()

  if not valid_contours:
    return None, None, mask, annotated

  largest = max(valid_contours, key=cv2.contourArea)

  M = cv2.moments(largest)
  if M["m00"] == 0:
    return None, None, mask, annotated

  cx = int(M["m10"] / M["m00"]) + roi_offset_x
  cy = int(M["m01"] / M["m00"]) + roi_offset_y

  cv2.circle(annotated, (cx, cy), 8, draw_color, -1)
  return cx, cy, mask, annotated


def save_calibration_frames(
    video_path,
    output_dir,
    frame_index=0,
    green_to_red_m=0.1507,
    sigma_pos_px=2.0,
    sigma_L2_m=0.001,
):
  """Slaat kalibratieafbeeldingen op en berekent px/cm met onzekerheid sigma_px_cm."""
  output_dir = Path(output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)

  cap = cv2.VideoCapture(str(video_path))
  cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
  success, frame = cap.read()
  cap.release()

  if not success or frame is None:
    raise ValueError(
        f"Kon frame {frame_index} niet inlezen van {video_path.name}."
    )

  # Versoepelde HSV grenzen voor rood en groen
  red_lowers = [np.array([0, 80, 50]), np.array([170, 80, 50])]
  red_uppers = [np.array([10, 255, 255]), np.array([180, 255, 255])]

  green_lowers = [np.array([35, 50, 40])]
  green_uppers = [np.array([85, 255, 255])]

  # Ruimere oppervlaktegrens (min_area=5, max_area=5000)
  rx, ry, _, annotated_red = detect_color_blob(
      frame,
      red_lowers,
      red_uppers,
      draw_color=(255, 0, 0),
      min_area=5,
      max_area=5000,
  )
  gx, gy, _, annotated_green = detect_color_blob(
      frame,
      green_lowers,
      green_uppers,
      draw_color=(0, 0, 255),
      min_area=5,
      max_area=5000,
  )

  cv2.imwrite(
      str(output_dir / f"{video_path.stem}_calibration_red.png"),
      annotated_red,
  )
  cv2.imwrite(
      str(output_dir / f"{video_path.stem}_calibration_green.png"),
      annotated_green,
  )

  if rx is None or gx is None:
    raise ValueError(
        f"Kalibratie mislukt op frame {frame_index}: Rood ({rx}) of groen"
        f" ({gx}) punt niet gevonden."
    )

  dx = gx - rx
  dy = gy - ry
  D_pixels = np.sqrt(dx**2 + dy**2)

  pixels_per_meter = D_pixels / green_to_red_m
  sigma_D_pixels = np.sqrt(2) * sigma_pos_px
  rel_err_sq = (sigma_D_pixels / D_pixels) ** 2 + (
      sigma_L2_m / green_to_red_m
  ) ** 2
  sigma_ppm = pixels_per_meter * np.sqrt(rel_err_sq)

  px_per_cm = pixels_per_meter / 100.0
  sigma_px_cm = sigma_ppm / 100.0

  return px_per_cm, sigma_px_cm, (rx, ry, gx, gy)