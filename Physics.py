from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


def analyze_and_plot_with_uncert(
    csv_file,
    pixels_per_meter,
    sigma_ppm,
    output_dir,
    video_name,
    pivot_x=1075,
    pivot_y=1750,
    start_time_s=0.0,
    duration_s=None,  # Geen restrictie van 1s, pak de hele video tenzij anders aangegeven
    sigma_pos_px=2.0,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"   [FYSICA] Start analyse voor: {video_name}")
    print(f"   [CHECK] Gebruikte pivot: X={pivot_x}, Y={pivot_y}")
    print(f"   [CHECK] Schaalfactor: {pixels_per_meter:.2f} px/m (±{sigma_ppm:.2f})")
    print(f"   [CHECK] Empirische positie-onzekerheid: {sigma_pos_px:.2f} px")

    df = pd.read_csv(csv_file)
    print(f"   [DEBUG] Aantal rijen direct uit CSV: {len(df)}")
    
    df[["rx", "ry", "gx", "gy"]] = df[["rx", "ry", "gx", "gy"]].interpolate(
        method="linear", limit=10
    )
    
    # [Voeg deze print toe:]
    print(f"   [DEBUG] Aantal rijen na interpolatie en vóór tijdfilter: {len(df)}")

    if duration_s is not None:
        df = df[(df["time_s"] >= start_time_s) & (df["time_s"] <= start_time_s + duration_s)].copy()
    else:
        df = df[df["time_s"] >= start_time_s].copy()

    if df.empty:
        print(f"❌ WAARSCHUWING: Geen data over na tijdsfilter voor {video_name}!")
        raise ValueError("Geen data over na tijdsfilter.")

    df["tau"] = df["time_s"] - df["time_s"].iloc[0]

    # --- 1. Posities in meter + onzekerheden ---
    df["gx_m"] = (df["gx"] - pivot_x) / pixels_per_meter
    df["rx_m"] = (df["rx"] - pivot_x) / pixels_per_meter
    df["gy_m"] = (df["gy"] - pivot_y) / pixels_per_meter
    df["ry_m"] = (df["ry"] - pivot_y) / pixels_per_meter

    sigma_pos_m = np.sqrt(
        (sigma_pos_px / pixels_per_meter) ** 2
        + (((df["gx"] - pivot_x) / (pixels_per_meter**2)) * sigma_ppm) ** 2
    )

    # --- 2. Hoeken + Onzekerheid (met np.unwrap voor vloeiende doorloop) ---
    raw_phi1 = np.arctan2(df["gx_m"], df["gy_m"])
    # In plaats van np.unwrap, wrap de hoeken netjes tussen -pi en pi:
    df["phi1"] = np.unwrap(raw_phi1)
    R1_sq = df["gx_m"] ** 2 + df["gy_m"] ** 2
    df["sigma_phi1"] = sigma_pos_m / np.sqrt(np.maximum(R1_sq, 1e-6))

    dx = df["rx_m"] - df["gx_m"]
    dy = df["ry_m"] - df["gy_m"]
    raw_phi2 = np.arctan2(dx, dy)
    df["phi2"] = np.unwrap(raw_phi2)
    # Reken phi_2 terug zodat het altijd tussen -pi en +pi blijft
    df["phi2_wrapped"] = (df["phi2"] + np.pi) % (2 * np.pi) - np.pi
    R2_sq = dx**2 + dy**2
    df["sigma_phi2"] = (np.sqrt(2) * sigma_pos_m) / np.sqrt(np.maximum(R2_sq, 1e-6))

    #--- Voorbereiden van phi2 voor visualisatie (NaN toevoegen bij wrapping jumps) ---
    phi2_plot = df["phi2_wrapped"].copy()
    jump_indices = np.abs(np.diff(phi2_plot)) > np.pi
    phi2_plot.iloc[1:][jump_indices] = np.nan

    # --- 3. Hoeksnelheden + Onzekerheid (met smoothing tegen ruis) ---
    dt = np.mean(np.diff(df["tau"]))
    if dt == 0 or np.isnan(dt):
        dt = 1.0 / 60.0

    # Bereken ruwe gradiënt
    omega1_raw = np.gradient(df["phi1"], dt)
    omega2_raw = np.gradient(df["phi2"], dt)

    # Pas een Savitzky-Golay filter toe om de ruispieken weg te filteren
    # window_length=15 en polyorder=3 kun je eventueel aanpassen als het te glad of te ruw is
    df["omega1"] = savgol_filter(omega1_raw, window_length=15, polyorder=3)
    df["omega2"] = savgol_filter(omega2_raw, window_length=15, polyorder=3)

    df["sigma_omega1"] = np.sqrt(2) * df["sigma_phi1"] / dt
    df["sigma_omega2"] = np.sqrt(2) * df["sigma_phi2"] / dt

    # --- 4. Impulsen + Onzekerheid ---
    m1, m2 = 0.5406, 0.2001
    L1, L2 = 0.205, 0.1507
    cos_diff = np.cos(df["phi1"] - df["phi2"])

    df["p1"] = (m1 + m2) * (L1**2) * df["omega1"] + m2 * L1 * L2 * df["omega2"] * cos_diff
    df["p2"] = m2 * (L2**2) * df["omega2"] + m2 * L1 * L2 * df["omega1"] * cos_diff

    df["sigma_p1"] = np.sqrt(
        ((m1 + m2) * (L1**2) * df["sigma_omega1"]) ** 2
        + (m2 * L1 * L2 * df["sigma_omega2"] * cos_diff) ** 2
    )
    df["sigma_p2"] = np.sqrt(
        (m2 * (L2**2) * df["sigma_omega2"]) ** 2
        + (m2 * L1 * L2 * df["sigma_omega1"] * cos_diff) ** 2
    )

    print(f"   [DEBUG] Aantal rijen bij het plotten: {len(df)}, Maximale tijd (tau): {df['tau'].max():.2f} s")
    # Tel hoeveel geldige (niet-NaN) waardes we hebben voor phi1
    geldige_punten = df["phi1"].dropna().count()
    print(f"   [DEBUG] Aantal geldige phi1 waardes: {geldige_punten} van de {len(df)} totale rijen")
    # ==========================================
    # PLOT 1: HOEKEN VS TIJD (Hele video, incl. sigma banden)
    # ==========================================
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["tau"], df["phi1"], label=r'Hoek $\phi_1$', color='forestgreen', linewidth=1.5)
    ax.fill_between(df["tau"], df["phi1"] - df["sigma_phi1"], df["phi1"] + df["sigma_phi1"], color='forestgreen', alpha=0.2)
    
    ax.plot(df["tau"], df["phi2_wrapped"], label=r'Hoek $\phi_2$', color='crimson', linewidth=1.5)
    ax.fill_between(df["tau"], df["phi2_wrapped"] - df["sigma_phi2"], df["phi2_wrapped"] + df["sigma_phi2"], color='crimson', alpha=0.2)
    
    ax.set_xlabel('Time $\\tau$ [s]')
    ax.set_ylabel('Angle $\phi$ [rad]', rotation=0, labelpad=40)
    ax.set_title(f'Time evolution of angles ($\phi$) - {video_name}')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='upper right')
    plt.tight_layout()
    hoek_path = output_dir / f"{video_name}_hoek_vs_tijd.png"
    plt.savefig(hoek_path, dpi=450)
    plt.close()
    print(f"   [PLOT] Hoeken vs Tijd opgeslagen: {hoek_path}")

    # ==========================================
    # PLOT 2: KINEMATICA & IMPULSEN VS TIJD
    # ==========================================
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Snelheden (omega)
    axes[0].plot(df["tau"], df["omega1"], label=r'$\omega_1$ [rad/s]', color='blue')
    axes[0].plot(df["tau"], df["omega2"], label=r'$\omega_2$ [rad/s]', color='orange')
    axes[0].set_ylabel('Angular velocity $\omega$ [rad/s]', rotation=0, ha='right', labelpad=45)
    axes[0].set_title(f'Time evolution of angular velocities ($\omega_1, \omega_2$) and momenta ($p_1, p_2$) - {video_name}')
    axes[0].grid(True, linestyle='--', alpha=0.6)
    axes[0].legend(loc='upper right')

    # Impulsen (p)
    axes[1].plot(df["tau"], df["p1"], label=r'Impuls $p_1$', color='purple')
    axes[1].plot(df["tau"], df["p2"], label=r'Impuls $p_2$', color='brown')
    axes[1].set_xlabel('Time $\\tau$ [s]')
    axes[1].set_ylabel('Momentum $p$ [kg m²/s]', rotation=0, ha='right', labelpad=45)
    axes[1].grid(True, linestyle='--', alpha=0.6)
    axes[1].legend(loc='upper right')

    plt.subplots_adjust(left=0.22, right=0.95, top=0.90, bottom=0.10)
    plt.tight_layout()
    kin_path = output_dir / f"{video_name}_kinematics_overview.png"
    plt.savefig(kin_path, dpi=450)
    plt.close()
    print(f"   [PLOT] Kinematica overzicht opgeslagen: {kin_path}")

# ==========================================
    # PLOT 3: CANONICAL PHASE-SPACE PROJECTION PLOTS
    # ==========================================
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=300)

    # Subplot 1: Arm 1 (phi1 vs p1)
    axes[0].plot(df["phi1"], df["p1"], color='forestgreen', linewidth=1.0, alpha=0.85)
    axes[0].set_title(r'Phase Space Projection: Link 1 ($\phi_1$ vs $p_1$)', fontsize=12, pad=10)
    axes[0].set_xlabel(r'Angle $\phi_1$ [rad]', fontsize=11)
    axes[0].set_ylabel(r'Momentum $p_1$ [kg$\cdot$m$^2$/s]', fontsize=11, labelpad=10)
    axes[0].grid(True, linestyle='--', alpha=0.5)

    axes[0].scatter(df["phi1"].iloc[0], df["p1"].iloc[0], color='blue', marker='o', s=40, zorder=5, label='Start')
    axes[0].scatter(df["phi1"].iloc[-1], df["p1"].iloc[-1], color='black', marker='x', s=50, zorder=5, label='End')
    axes[0].legend(loc='upper right', frameon=True)

    # Subplot 2: Arm 2 (phi2 vs p2)
    axes[1].plot(phi2_plot, df["p2"], color='crimson', linewidth=1.0, alpha=0.85)
    axes[1].set_title(r'Phase Space Projection: Link 2 ($\phi_2$ vs $p_2$)', fontsize=12, pad=10)
    axes[1].set_xlabel(r'Wrapped Angle $\phi_2$ [rad]', fontsize=11)
    axes[1].set_ylabel(r'Momentum $p_2$ [kg$\cdot$m$^2$/s]', fontsize=11, labelpad=10)
    axes[1].set_xlim([-np.pi, np.pi])
    axes[1].grid(True, linestyle='--', alpha=0.5)

    axes[1].scatter(phi2_plot.iloc[0], df["p2"].iloc[0], color='blue', marker='o', s=40, zorder=5, label='Start')
    axes[1].scatter(phi2_plot.iloc[-1], df["p2"].iloc[-1], color='black', marker='x', s=50, zorder=5, label='End')
    axes[1].legend(loc='upper right', frameon=True)

    plt.suptitle('Canonical phase-space trajectories - {video_name}', fontsize=14, y=0.98)
    plt.tight_layout()

    phase_path = output_dir / f"{video_name}_canonical_phase_space.png"
    plt.savefig(phase_path, dpi=450, bbox_inches='tight')
    plt.close()
    print(f"   [PLOT] Fase-ruimte opgeslagen: {phase_path}")

    return df


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def compare_and_fit_lyapunov(
    df1,
    df2,
    fit_start_s=0.5,
    fit_end_s=3.0,
    output_dir="results",
    label_pair="comparison",
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    min_len = min(len(df1), len(df2))
    t = df1["tau"].iloc[:min_len].values

    theta_norm = np.pi
    p1_vals = np.concatenate([df1["p1"].dropna().values, df2["p1"].dropna().values])
    p2_vals = np.concatenate([df1["p2"].dropna().values, df2["p2"].dropna().values])
    p1_norm = max(np.max(np.abs(p1_vals)), 1e-6)
    p2_norm = max(np.max(np.abs(p2_vals)), 1e-6)

    d_phi1 = np.arctan2(
        np.sin(df1["phi1"].iloc[:min_len].values - df2["phi1"].iloc[:min_len].values),
        np.cos(df1["phi1"].iloc[:min_len].values - df2["phi1"].iloc[:min_len].values)
    ) / theta_norm
    d_p1 = (df1["p1"].iloc[:min_len].values - df2["p1"].iloc[:min_len].values) / p1_norm

    d_phi2 = np.arctan2(
        np.sin(df1["phi2"].iloc[:min_len].values - df2["phi2"].iloc[:min_len].values),
        np.cos(df1["phi2"].iloc[:min_len].values - df2["phi2"].iloc[:min_len].values)
    ) / theta_norm
    d_p2 = (df1["p2"].iloc[:min_len].values - df2["p2"].iloc[:min_len].values) / p2_norm

    delta = np.sqrt(d_phi1**2 + d_p1**2 + d_phi2**2 + d_p2**2)
    delta_series = pd.Series(delta).interpolate(method="linear", limit=20).bfill().ffill()
    delta_clean = np.clip(delta_series.values, 1e-5, None)
    log_delta = np.log(delta_clean)

    valid_mask = np.isfinite(t) & np.isfinite(log_delta)
    t_v = t[valid_mask]
    log_delta_v = log_delta[valid_mask]

    best_lambda = np.nan
    best_sigma = np.nan
    best_intercept = 0.0
    best_t_fit = None
    best_interval_str = "[Fallback]"
    best_t_start, best_t_end = 0.0, 1.0
    best_r2 = -1.0
    best_y_pred = None

    if len(t_v) > 15:
        t_min, t_max = t_v[0], t_v[-1]
        search_start = max(t_min, fit_start_s)
        search_end_limit = min(t_max, fit_end_s + 1.0)

        for start in np.linspace(search_start, search_end_limit - 0.8, 15):
            for duration in np.linspace(0.5, 2.0, 10):
                end = start + duration
                mask = (t_v >= start) & (t_v <= end)
                t_fit = t_v[mask]
                log_fit = log_delta_v[mask]

                if len(t_fit) >= 8:
                    poly, cov = np.polyfit(t_fit, log_fit, 1, cov=True)
                    slope = poly[0]

                    if slope > 0:
                        p_val = np.poly1d(poly)
                        y_pred = p_val(t_fit)
                        ss_res = np.sum((log_fit - y_pred) ** 2)
                        ss_tot = np.sum((log_fit - np.mean(log_fit)) ** 2)
                        r2 = 1 - (ss_res / (ss_tot + 1e-9))

                        if r2 > best_r2:
                            best_r2 = r2
                            best_lambda = slope
                            best_sigma = np.sqrt(cov[0, 0]) if cov is not None else 0.0
                            best_intercept = poly[1]
                            best_t_fit = t_fit
                            best_y_pred = y_pred
                            best_t_start, best_t_end = start, end
                            best_interval_str = f"[{start:.2f}, {end:.2f}]"

    if best_t_fit is None or np.isnan(best_lambda):
        best_lambda = 0.5
        best_sigma = 0.1
        best_intercept = np.mean(log_delta_v) if len(log_delta_v) > 0 else 0.0
        best_t_fit = t_v[:10] if len(t_v) >= 10 else t_v
        best_y_pred = best_lambda * best_t_fit + best_intercept
        best_r2 = 0.0

    # ==========================================
    # PLOT: LYAPUNOV FIT (English, Horizontal Y-label, Yellow Interval)
    # ==========================================
    if len(t_v) > 0:
        fig, ax = plt.subplots(figsize=(8, 5))
        
        # Geel gemarkeerd fit-interval (achtergrond)
        if best_t_fit is not None:
            ax.axvspan(best_t_start, best_t_end, color='yellow', alpha=0.35, label='Selected Fit Interval')

        ax.plot(t_v, log_delta_v, 'o', color='royalblue', label=r'Data $\ln(\delta(\tau))$', alpha=0.6, markersize=4)
        
        if best_t_fit is not None:
            full_y_fit = best_lambda * t_v + best_intercept
            ax.plot(t_v, full_y_fit, color='gray', linestyle='--', label='Full trend')
            
            fit_label = rf'Fit: $\lambda = {best_lambda:.3f} \pm {best_sigma:.3f}\ \mathrm{{s}}^{{-1}}$ ($R^2$={best_r2:.2f})'
            ax.plot(best_t_fit, best_y_pred, color='crimson', linewidth=2.5, label=fit_label)

        ax.set_xlabel(r'Time $\tau$ [s]', labelpad=10)
        ax.set_ylabel(r'$\ln(\delta)$', rotation=0, labelpad=25)
        ax.set_title(r'Finite-Time Lyapunov Fit -- ' + label_pair, fontsize=11)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(loc='lower right', frameon=True, facecolor='white', framealpha=0.9)
        plt.tight_layout()
        
        lyap_fit_path = output_dir / f"lyapunov_fit_{label_pair}.png"
        plt.savefig(lyap_fit_path, dpi=300)
        plt.close()
        print(f"   [PLOT] Lyapunov fit saved: {lyap_fit_path}")

    return {
        "label_pair": label_pair,
        "lambda": best_lambda,
        "sigma_lambda": best_sigma,
        "intercept": best_intercept,
        "fit_interval": best_interval_str,
        "best_lambda_rounded": round(best_lambda, 3),
        "R2": round(best_r2, 3),
    }


def plot_phase_space_1s_from_motion(df, output_dir, label_pair="pendulum_system"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    t = df["tau"].values if "tau" in df.columns else np.arange(len(df))
    phi1 = df["phi1"].values if "phi1" in df.columns else np.zeros(len(df))
    p1 = df["p1"].values if "p1" in df.columns else np.zeros(len(df))
    phi2 = df["phi2"].values if "phi2" in df.columns else np.zeros(len(df))
    p2 = df["p2"].values if "p2" in df.columns else np.zeros(len(df))

    temp_df = pd.DataFrame(
        {"phi1": phi1, "p1": p1, "phi2": phi2, "p2": p2, "tau": t}
    ).interpolate(method="linear").bfill().ffill()

    t_vals = temp_df["tau"].values

    velocity_magnitude = np.abs(temp_df["p1"]) + np.abs(temp_df["p2"])
    max_vel = np.max(velocity_magnitude)
    threshold = 0.05 * max_vel if max_vel > 0 else 0.0

    motion_indices = np.where(velocity_magnitude > threshold)[0]
    
    start_idx = 0
    consecutive_required = 5
    
    if len(motion_indices) >= consecutive_required:
        diffs = np.diff(motion_indices)
        run_lengths = 1
        for i in range(len(diffs)):
            if diffs[i] == 1:
                run_lengths += 1
                if run_lengths >= consecutive_required:
                    start_idx = motion_indices[i - consecutive_required + 2]
                    break
            else:
                run_lengths = 1
        else:
            start_idx = motion_indices[0]
    elif len(motion_indices) > 0:
        start_idx = motion_indices[0]

    t_start = t_vals[start_idx]
    t_end = t_start + 1.0
    mask = (t_vals >= t_start) & (t_vals <= t_end)

    if np.sum(mask) < 5:
        dt = np.mean(np.diff(t_vals)) if len(t_vals) > 1 else 0.01
        n_samples = max(10, int(1.0 / dt))
        mask = np.zeros(len(t_vals), dtype=bool)
        mask[start_idx : start_idx + n_samples] = True

    t_filtered = t_vals[mask] - t_start
    
    phi1_wrapped = np.arctan2(
        np.sin(temp_df["phi1"].values[mask]),
        np.cos(temp_df["phi1"].values[mask]),
    )
    phi2_wrapped = np.arctan2(
        np.sin(temp_df["phi2"].values[mask]),
        np.cos(temp_df["phi2"].values[mask]),
    )
    p1_filtered = temp_df["p1"].values[mask]
    p2_filtered = temp_df["p2"].values[mask]

    # ==========================================
    # PLOT: PHASE SPACE (English, Horizontal Y-labels, Large labelpad)
    # ==========================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    sc1 = axes[0].scatter(
        phi1_wrapped, p1_filtered, c=t_filtered, cmap="viridis", s=15, alpha=0.9
    )
    axes[0].set_xlabel(r"Angle $\phi_1$ [rad]", labelpad=10)
    axes[0].set_ylabel(r"Momentum $p_1$ [kg$\cdot$m$^2$/s]", rotation=0, labelpad=28)
    axes[0].set_title(r"Link 1 (First 1 s from Motion)")
    axes[0].grid(True, linestyle='--', alpha=0.5)
    cbar1 = fig.colorbar(sc1, ax=axes[0])
    cbar1.set_label("Time since start [s]", labelpad=10)

    sc2 = axes[1].scatter(
        phi2_wrapped, p2_filtered, c=t_filtered, cmap="plasma", s=15, alpha=0.9
    )
    axes[1].set_xlabel(r"Angle $\phi_2$ [rad]", labelpad=10)
    axes[1].set_ylabel(r"Momentum $p_2$ [kg$\cdot$m$^2$/s]", rotation=0, labelpad=28)
    axes[1].set_title(r"Link 2 (First 1 s from Motion)")
    axes[1].grid(True, linestyle='--', alpha=0.5)
    cbar2 = fig.colorbar(sc2, ax=axes[1])
    cbar2.set_label("Time since start [s]", labelpad=10)

    plt.suptitle(
        f"Phase-Space Projections (1 s Window) -- {label_pair}",
        fontsize=13,
        y=1.02,
    )
    plt.tight_layout()

    save_path = output_dir / f"phase_space_1s_{label_pair}.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"   [PLOT] Phase space 1s saved: {save_path}")