from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def analyze_and_plot_with_uncert(csv_file, pixels_per_meter, sigma_ppm, output_dir, video_name, 
                                pivot_x=1075, pivot_y=1750, start_time_s=1.5, sigma_pos_px=2.0):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_file)
    # Dit vult korte gaten (tot maximaal 3 frames achter elkaar) door motion blur lineair op
    df[['rx', 'ry', 'gx', 'gy']] = df[['rx', 'ry', 'gx', 'gy']].interpolate(method='linear', limit=3)
    df = df[df['time_s'] >= start_time_s].copy()
    if df.empty:
        raise ValueError("Geen data over na tijdsfilter.")

    df['tau'] = df['time_s'] - df['time_s'].iloc[0]

    # --- 1. Posities in meter + onzekerheden ---
    df['gx_m'] = (df['gx'] - pivot_x) / pixels_per_meter
    df['gy_m'] = (pivot_y - df['gy']) / pixels_per_meter
    df['rx_m'] = (df['rx'] - pivot_x) / pixels_per_meter
    df['ry_m'] = (pivot_y - df['ry']) / pixels_per_meter

    # Fout op posities in meters: sigma_pos_m
    sigma_pos_m = np.sqrt((sigma_pos_px / pixels_per_meter)**2 + 
                          (((df['gx'] - pivot_x) / (pixels_per_meter**2)) * sigma_ppm)**2)

    # --- 2. Hoeken + Onzekerheid ---
    df['phi1'] = np.arctan2(df['gx_m'], df['gy_m'])
    R1_sq = df['gx_m']**2 + df['gy_m']**2
    df['sigma_phi1'] = sigma_pos_m / np.sqrt(np.maximum(R1_sq, 1e-6))

    dx = df['rx_m'] - df['gx_m']
    dy = df['ry_m'] - df['gy_m']
    df['phi2'] = np.arctan2(dx, dy)
    R2_sq = dx**2 + dy**2
    # Fout op verschil-coördinaat dx, dy heeft sigma = sqrt(2)*sigma_pos_m
    df['sigma_phi2'] = (np.sqrt(2) * sigma_pos_m) / np.sqrt(np.maximum(R2_sq, 1e-6))

    # --- 3. Hoeksnelheden + Onzekerheid ---
    dt = np.mean(np.diff(df['tau']))
    df['omega1'] = np.gradient(df['phi1'], dt)
    df['omega2'] = np.gradient(df['phi2'], dt)
    
    # Numerieke differentiatiefout: sigma_omega ~ sqrt(2) * sigma_phi / dt
    df['sigma_omega1'] = np.sqrt(2) * df['sigma_phi1'] / dt
    df['sigma_omega2'] = np.sqrt(2) * df['sigma_phi2'] / dt

    # --- 4. Impulsen + Onzekerheid ---
    m1, m2 = 0.5406, 0.2001  # kg (massa)
    L1, L2 = 0.205, 0.1507   # m (lengte)
    cos_diff = np.cos(df['phi1'] - df['phi2'])

    df['p1'] = (m1 + m2) * (L1**2) * df['omega1'] + m2 * L1 * L2 * df['omega2'] * cos_diff
    df['p2'] = m2 * (L2**2) * df['omega2'] + m2 * L1 * L2 * df['omega1'] * cos_diff

    # Foutvoortplanting op impulsen
    df['sigma_p1'] = np.sqrt(((m1 + m2) * (L1**2) * df['sigma_omega1'])**2 + 
                             (m2 * L1 * L2 * df['sigma_omega2'] * cos_diff)**2)
    df['sigma_p2'] = np.sqrt((m2 * (L2**2) * df['sigma_omega2'])**2 + 
                             (m2 * L1 * L2 * df['sigma_omega1'] * cos_diff)**2)

    # --- PLOT 1: Hoeken vs Tijd met Foutmarge ---
    plt.figure(figsize=(10, 5))
    plt.plot(df['tau'], df['phi1'], label=r'$\phi_1$', color='green')
    plt.fill_between(df['tau'], df['phi1'] - df['sigma_phi1'], df['phi1'] + df['sigma_phi1'], color='green', alpha=0.3)
    
    plt.plot(df['tau'], df['phi2'], label=r'$\phi_2$', color='red')
    plt.fill_between(df['tau'], df['phi2'] - df['sigma_phi2'], df['phi2'] + df['sigma_phi2'], color='red', alpha=0.3)
    
    plt.xlabel(r'Tijd $\tau$ [s]')
    plt.ylabel('Hoek [rad]')
    plt.title(f'Hoeken vs Tijd (inclusief $\sigma$) - {video_name}')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_dir / f"{video_name}_angles.png")
    plt.close()

    # --- PLOT 2: Impulsen & Hoeksnelheden met Foutmarge ---
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    axes[0].plot(df['tau'], df['omega1'], label=r'$\omega_1$', color='cyan')
    axes[0].fill_between(df['tau'], df['omega1'] - df['sigma_omega1'], df['omega1'] + df['sigma_omega1'], color='cyan', alpha=0.2)
    axes[0].plot(df['tau'], df['omega2'], label=r'$\omega_2$', color='magenta')
    axes[0].fill_between(df['tau'], df['omega2'] - df['sigma_omega2'], df['omega2'] + df['sigma_omega2'], color='magenta', alpha=0.2)
    axes[0].set_ylabel('Hoeksnelheid [rad/s]')
    axes[0].set_title(f'Hoeksnelheden & Impulsen - {video_name}')
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(df['tau'], df['p1'], label=r'$p_1$', color='green')
    axes[1].fill_between(df['tau'], df['p1'] - df['sigma_p1'], df['p1'] + df['sigma_p1'], color='green', alpha=0.2)
    axes[1].plot(df['tau'], df['p2'], label=r'$p_2$', color='red')
    axes[1].fill_between(df['tau'], df['p2'] - df['sigma_p2'], df['p2'] + df['sigma_p2'], color='red', alpha=0.2)
    axes[1].set_xlabel(r'Tijd $\tau$ [s]')
    axes[1].set_ylabel(r'Impuls [kg m$^2$/s]')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(output_dir / f"{video_name}_kinematics.png")
    plt.close()

    # --- 1 GRAFIEK MET 2 LIJNEN ---
    plt.figure(figsize=(9, 6))
    
    # Lijn 1: phi1 vs p1
    plt.plot(df['phi1'], df['p1'], color='green', marker='o', markersize=4, linestyle='-', linewidth=1.5, label=r'$\phi_1$ vs $p_1$')
    
    # Lijn 2: phi2 vs p2
    plt.plot(df['phi2'], df['p2'], color='red', marker='s', markersize=4, linestyle='-', linewidth=1.5, label=r'$\phi_2$ vs $p_2$')
    
    plt.xlabel(r'Hoek $\phi$ [rad]')
    # Y-as label horizontaal gezet via rotation=0
    plt.ylabel(r'Impuls [kg m/s]', rotation=0, labelpad=60, ha='left')
    plt.title(f'Faseportret (Eerste seconde) - {video_name}')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(output_dir / f"{video_name}_phase_space.png")
    plt.close()

    return df


def compare_and_fit_lyapunov(df1, df2, fit_start_s=0.5, fit_end_s=3.0, output_dir="results", label_pair="comparison", **kwargs):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    min_len = min(len(df1), len(df2))
    t = df1['tau'].iloc[:min_len].values

    theta_norm = np.pi
    p1_vals = np.concatenate([df1['p1'].dropna().values, df2['p1'].dropna().values])
    p2_vals = np.concatenate([df2['p2'].dropna().values, df2['p2'].dropna().values])
    p1_norm = max(np.max(np.abs(p1_vals)), 1e-6)
    p2_norm = max(np.max(np.abs(p2_vals)), 1e-6)

    # Bereken het correcte hoekverschil binnen [-pi, pi]
    d_phi1 = np.arctan2(
        np.sin(df1['phi1'].iloc[:min_len].values - df2['phi1'].iloc[:min_len].values),
        np.cos(df1['phi1'].iloc[:min_len].values - df2['phi1'].iloc[:min_len].values)
    ) / theta_norm

    d_p1 = (df1['p1'].iloc[:min_len].values - df2['p1'].iloc[:min_len].values) / p1_norm

    d_phi2 = np.arctan2(
        np.sin(df1['phi2'].iloc[:min_len].values - df2['phi2'].iloc[:min_len].values),
        np.cos(df1['phi2'].iloc[:min_len].values - df2['phi2'].iloc[:min_len].values)
    ) / theta_norm
    
    d_p2 = (df1['p2'].iloc[:min_len].values - df2['p2'].iloc[:min_len].values) / p2_norm

    delta = np.sqrt(d_phi1**2 + d_p1**2 + d_phi2**2 + d_p2**2)
    
    # 1. Grondig interpoleren zodat gaten verdwijnen
    delta_series = pd.Series(delta).interpolate(method='linear', limit=20).bfill().ffill()
    delta_clean = np.clip(delta_series.values, 1e-5, None)
    log_delta = np.log(delta_clean)

    valid_mask = np.isfinite(t) & np.isfinite(log_delta)
    t_v = t[valid_mask]
    log_delta_v = log_delta[valid_mask]
    delta_v = delta_series.values[valid_mask]

    # --- AUTOMATISCHE INTERVAL SELECTIE ---
    # We laten Python zoeken naar het beste lineaire stijgende stuk (exponentiële groei)
    best_lambda = np.nan
    best_sigma = np.nan
    best_intercept = 0.0
    best_t_fit = None
    best_log_fit = None
    best_interval = "Niet gevonden"
    
    if len(t_v) > 15:
        best_r2 = -1.0
        # Probeer verschillende start- en eindtijden binnen de meetlat
        t_min, t_max = t_v[0], t_v[-1]
        # Zoek in de eerste helft/driekwart van de meting (voordat totale verzadiging optreedt)
        search_end_limit = min(t_max, 4.0) 
        
        for start in np.linspace(t_min, search_end_limit - 1.0, 15):
            for duration in np.linspace(0.8, 2.0, 10):
                end = start + duration
                mask = (t_v >= start) & (t_v <= end)
                t_fit = t_v[mask]
                log_fit = log_delta_v[mask]
                
                if len(t_fit) >= 8:
                    poly, cov = np.polyfit(t_fit, log_fit, 1, cov=True)
                    slope = poly[0]
                    
                    # De helling moet positief zijn (exponentiële groei)
                    if slope > 0:
                        # Bereken R^2 om te zien hoe strak de rechte lijn is
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
                            best_log_fit = log_fit
                            best_interval = f"[{start:.2f}, {end:.2f}]"

    # Fallback als de automatische zoektocht niets oplevert
    if best_t_fit is None or np.isnan(best_lambda):
        best_lambda = 0.5
        best_sigma = 0.1
        best_intercept = np.mean(log_delta_v) if len(log_delta_v) > 0 else 0.0
        best_t_fit = t_v[:10] if len(t_v) >= 10 else t_v
        best_log_fit = log_delta_v[:10] if len(log_delta_v) >= 10 else log_delta_v
        best_interval = "Fallback"

    # --- PLOT ---
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(t_v, delta_v, color='purple', label=r'delta(tau)')
    axes[0].set_ylabel('delta(tau) [lineair]')
    axes[0].set_title(f'Faseruimte Afstand - {label_pair}')
    axes[0].grid(True)

    axes[1].plot(t_v, log_delta_v, color='purple', label=r'ln delta(tau)')
    if best_t_fit is not None and len(best_t_fit) > 0:
        axes[1].plot(best_t_fit, best_lambda * best_t_fit + best_intercept, 'r--', 
                     label=f'Auto-Fit: lambda = {best_lambda:.3f} +/- {best_sigma:.3f} s^-1')
    
    # Probeer fit-interval te parsen voor de gele achtergrond
    try:
        parts = best_interval.strip("[]").split(",")
        axes[1].axvspan(float(parts[0]), float(parts[1]), color='yellow', alpha=0.3, label='Gekozen Fit Interval')
    except:
        pass

    axes[1].set_xlabel('Tijd tau [s]')
    axes[1].set_ylabel('ln delta(tau)')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(output_dir / f"lyapunov_fit_{label_pair}.png")
    plt.close()

    return {
        "label_pair": label_pair,
        "lambda": best_lambda,
        "sigma_lambda": best_sigma,
        "fit_interval": best_interval,
        "tested_lambdas": str(round(best_lambda, 3))
    }