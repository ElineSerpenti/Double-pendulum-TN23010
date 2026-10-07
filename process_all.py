import itertools
from pathlib import Path
import cv2
import numpy as np
import pandas as pd

from Physics import analyze_and_plot_with_uncert, compare_and_fit_lyapunov
from Tracking import detect_color_blob, save_calibration_frames

def process_single_video(
    video_path, output_csv, output_video_path, search_margin=100
):
  cap = cv2.VideoCapture(str(video_path))
  fps = cap.get(cv2.CAP_PROP_FPS)
  w_frame = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
  h_frame = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

  fourcc = cv2.VideoWriter_fourcc(*"mp4v")
  out_video = cv2.VideoWriter(
      str(output_video_path), fourcc, fps, (w_frame, h_frame)
  )

  # Versoepelde kleurgrenzen voor betere detectie bij beweging
  red_lowers = [np.array([0, 80, 50]), np.array([170, 80, 50])]
  red_uppers = [np.array([10, 255, 255]), np.array([180, 255, 255])]

  green_lowers = [np.array([35, 50, 40])]
  green_uppers = [np.array([85, 255, 255])]

  results = []
  frame_idx = 0
  prev_red = None
  prev_green = None

  while True:
    ret, frame = cap.read()
    if not ret:
      break

    # Rood (blauwe cirkel op de video)
    rx, ry, _, annotated_frame = detect_color_blob(
        frame,
        red_lowers,
        red_uppers,
        prev_pos=prev_red,
        search_margin=search_margin,
        draw_color=(255, 0, 0),
        min_area=5,
        max_area=5000,
    )

    # Groen (rode cirkel op de video)
    gx, gy, _, annotated_frame = detect_color_blob(
        annotated_frame,
        green_lowers,
        green_uppers,
        prev_pos=prev_green,
        search_margin=search_margin,
        draw_color=(0, 0, 255),
        min_area=5,
        max_area=5000,
    )

    if rx is not None:
      prev_red = (rx, ry)
    if gx is not None:
      prev_green = (gx, gy)

    out_video.write(annotated_frame)

    results.append({
        "frame": frame_idx,
        "time_s": frame_idx / fps if fps > 0 else np.nan,
        "rx": rx,
        "ry": ry,
        "gx": gx,
        "gy": gy,
    })
    frame_idx += 1

  cap.release()
  out_video.release()

  output_csv = Path(output_csv)
  output_csv.parent.mkdir(parents=True, exist_ok=True)
  pd.DataFrame(results).to_csv(output_csv, index=False)


def run_pipeline():
  base_video_dir = Path("Video's")
  base_csv_dir = Path("csv-files")
  base_results_dir = Path("results")

  all_folders = sorted([f for f in base_video_dir.iterdir() if f.is_dir()])
  condition_folders = all_folders[:1]       # pakt even alleen befginvoorwaarde 1

  if not condition_folders:
    print(f"Geen submappen gevonden in {base_video_dir.resolve()}")
    return

  for cond_folder in condition_folders:
    cond_name = cond_folder.name
    print(f"\n==================================================")
    print(f" STARTEN MET MAP: {cond_name}")
    print(f"==================================================")

    cond_csv_dir = base_csv_dir / cond_name
    cond_results_dir = base_results_dir / cond_name
    cond_csv_dir.mkdir(parents=True, exist_ok=True)
    cond_results_dir.mkdir(parents=True, exist_ok=True)

    video_files = sorted(
        [v for v in cond_folder.iterdir() if v.suffix.lower() in [".mp4", ".mov"]]
    )

    if not video_files:
      print(f"Geen video's gevonden in {cond_folder}")
      continue

    analyzed_data = {}
    calibration_summary = []

    for video_path in video_files:
      video_name = video_path.stem
      print(f"\n--- Verwerken van video: {video_name} ---")

      raw_csv = cond_csv_dir / f"{video_name}.csv"
      video_out_dir = cond_results_dir / video_name
      video_out_dir.mkdir(parents=True, exist_ok=True)
      tracked_video_path = video_out_dir / f"{video_name}_tracked.mp4"

      if not raw_csv.exists() or not tracked_video_path.exists():
        print("1. Beeldverwerking/tracking starten incl video-export...")
        process_single_video(video_path, output_csv=raw_csv,
                  output_video_path=tracked_video_path)
      else:
        print(f"1. Bestaande CSV en getracktre video gebruikt: {video_name}")

      print("2. Kalibratieafbeeldingen opslaan & schaal berekenen...")
      try:
        # Vangt direct px_per_cm en sigma_px_cm op!
        px_per_cm, sigma_px_cm, _ = save_calibration_frames(
            video_path, output_dir=video_out_dir, frame_index=0
        )

        print(f"   - Kalibratie: {px_per_cm:.2f} ± {sigma_px_cm:.2f} px/cm")

        calibration_summary.append({
            "video": video_name,
            "px_per_cm": px_per_cm,
            "sigma_px_per_cm": sigma_px_cm,
        })

        print("3. Kinematische grafieken genereren...")
        # Omrekenen voor de natuurkunde-analyse (in meter)
        px_per_m = px_per_cm * 100.0
        sigma_ppm = sigma_px_cm * 100.0

        df_analyzed = analyze_and_plot_with_uncert(
            raw_csv,
            pixels_per_meter=px_per_m,
            sigma_ppm=sigma_ppm,
            output_dir=video_out_dir,
            video_name=video_name,
        )

        analyzed_data[video_name] = df_analyzed

      except Exception as e:
        print(f"❌ Fout bij verwerken van {video_name}: {e}")

    # Sla kalibratie overzicht op in px/cm
    pd.DataFrame(calibration_summary).to_csv(
        cond_results_dir / "kalibratie_overzicht.csv", index=False
    )

    # Paarsgewijze vergelijking per beginvoorwaarde
    print(f"\n--- Lyapunov vergelijkingen voor {cond_name} ---")
    video_keys = list(analyzed_data.keys())
    pairs = list(itertools.combinations(video_keys, 2))
    lyapunov_results = []

    lyapunov_dir = cond_results_dir / "lyapunov_vergelijkingen"
    lyapunov_dir.mkdir(exist_ok=True)

    for v1_name, v2_name in pairs:
      pair_label = f"{v1_name}_vs_{v2_name}"
      print(f"Vergelijken: {pair_label}")

      try:
        res = compare_and_fit_lyapunov(
            analyzed_data[v1_name],
            analyzed_data[v2_name],
            fit_start_s=0.5,
            fit_end_s=3.0,
            output_dir=lyapunov_dir,
            label_pair=pair_label,
        )
        lyapunov_results.append(res)
      except Exception as e:
        print(f"❌ Fout bij Lyapunov-fit van {pair_label}: {e}")

    if lyapunov_results:
      pd.DataFrame(lyapunov_results).to_csv(
          cond_results_dir / "lyapunov_exponenten.csv", index=False
      )

  print("\n✅ Alle beginvoorwaarden succesvol verwerkt!")


if __name__ == "__main__":
  run_pipeline()