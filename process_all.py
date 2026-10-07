from pathlib import Path
import pandas as pd
import cv2
import numpy as np

# Let op de hoofdletters P en T conform jouw bestandsnamen!
from Tracking import save_calibration_frames, detect_color_blob
from Physics import analyze_and_plot_with_uncert

def process_single_video(video_path, output_csv, search_margin=80):
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)

    red_lowers = [np.array([0, 80, 50]), np.array([170, 80, 50])]
    red_uppers = [np.array([10, 255, 255]), np.array([180, 255, 255])]
    green_lowers = [np.array([35, 50, 50])]
    green_uppers = [np.array([85, 255, 255])]

    results = []
    frame_idx = 0
    prev_red = None
    prev_green = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rx, ry, _, _ = detect_color_blob(frame, red_lowers, red_uppers, prev_pos=prev_red, search_margin=search_margin, draw_color=(255, 0, 0))
        gx, gy, _, _ = detect_color_blob(frame, green_lowers, green_uppers, prev_pos=prev_green, search_margin=search_margin, draw_color=(0, 0, 255))

        if rx is not None:
            prev_red = (rx, ry)
        if gx is not None:
            prev_green = (gx, gy)

        results.append({
            "frame": frame_idx,
            "time_s": frame_idx / fps if fps > 0 else np.nan,
            "rx": rx, "ry": ry,
            "gx": gx, "gy": gy
        })
        frame_idx += 1

    cap.release()
    pd.DataFrame(results).to_csv(output_csv, index=False)


if __name__ == "__main__":
    VIDEO_DIR = Path("Video's")
    RAW_CSV_DIR = Path("csv-files")
    PLOTS_DIR = Path("plots_and_calibrations")

    RAW_CSV_DIR.mkdir(exist_ok=True)
    PLOTS_DIR.mkdir(exist_ok=True)

    # Zoek alle .MOV videobestanden op
    video_files = sorted(list(VIDEO_DIR.glob("*.MOV")))

    if not video_files:
        print(f"Geen .MOV video's gevonden in {VIDEO_DIR.resolve()}")
    else:
        print(f"Gevonden video's: {[v.name for v in video_files]}\n")

    for video_path in video_files:
        print(f"==================================================")
        print(f" Verwerken van video: {video_path.name}")
        print(f"==================================================")
        
        raw_csv = RAW_CSV_DIR / f"{video_path.stem}.csv"
        video_output_dir = PLOTS_DIR / video_path.stem
        video_output_dir.mkdir(exist_ok=True)

        # 1. Video tracken naar CSV (als CSV nog niet bestaat)
        if not raw_csv.exists():
            print("Beeldverwerking/tracking starten...")
            process_single_video(video_path, raw_csv)
        else:
            print(f"Bestaand CSV-bestand gevonden: {raw_csv.name} (tracking overgeslagen)")

        # 2. Kalibratieafbeeldingen opslaan en schaalfactor berekenen op het EERSTE frame (frame_index=0)
        print("Kalibratieframe opslaan (frame index 0)...")
        try:
            px_per_m, sigma_ppm, _ = save_calibration_frames(
                video_path, 
                output_dir=video_output_dir, 
                frame_index=0
            )

            px_per_cm = px_per_m / 100.0
            sigma_px_per_cm = sigma_ppm / 100.0
            
            print(f"\n KALIBRATIE RESULTAAT ({video_path.name}):")
            print(f"  - Pixels per meter: {px_per_m:.2f} ± {sigma_ppm:.2f} px/m")
            print(f"  - Pixels per cm:    {px_per_cm:.2f} ± {sigma_px_per_cm:.2f} px/cm\n")

            # 3. Analyse & Grafieken genereren met foutmarge
            print("Grafieken voor hoeken, hoeksnelheden en impulsen genereren...")
            analyze_and_plot_with_uncert(
                raw_csv, 
                pixels_per_meter=px_per_m, 
                sigma_ppm=sigma_ppm,
                output_dir=video_output_dir, 
                video_name=video_path.stem
            )

            print(f"✓ Voltooid voor {video_path.name}! Resultaten in '{video_output_dir}'\n")

        except Exception as e:
            print(f"❌ Fout bij verwerken van {video_path.name}: {e}\n")

    print("Verwerking van alle video's voltooid!")