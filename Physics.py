from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def analyze_and_plot_with_uncert(csv_file, pixels_per_meter, sigma_ppm, output_dir, video_name, 
                                pivot_x=1075, pivot_y=1750, start_time_s=1.5, sigma_pos_px=2.0):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_file)
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

    return df


def compare_and_fit_lyapunov(df1, df2, fit_start_s, fit_end_s, output_dir, label_pair):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Bepaal het kleinste aantal frames tussen de twee video's
    min_len = min(len(df1), len(df2))
    
    # Koppel alle benodigde kolommen direct los als NumPy arrays met lengte min_len
    t = df1['tau'].iloc[:min_len].values

    theta_norm = np.pi
    p1_norm = max(np.max(np.abs(df1['p1'])), np.max(np.abs(df2['p1'])))
    p2_norm = max(np.max(np.abs(df1['p2'])), np.max(np.abs(df2['p2'])))

    # Verschillen berekenen
    d_phi1 = (df1['phi1'].iloc[:min_len].values - df2['phi1'].iloc[:min_len].values) / theta_norm
    d_p1 = (df1['p1'].iloc[:min_len].values - df2['p1'].iloc[:min_len].values) / p1_norm
    d_phi2 = (df1['phi2'].iloc[:min_len].values - df2['phi2'].iloc[:min_len].values) / theta_norm
    d_p2 = (df1['p2'].iloc[:min_len].values - df2['p2'].iloc[:min_len].values) / p2_norm

    delta = np.sqrt(d_phi1**2 + d_p1**2 + d_phi2**2 + d_p2**2)

    # Foutberekening (omgezet naar numpy arrays .values om lengte-fouten te voorkomen)
    sigma_d_phi1 = np.sqrt(df1['sigma_phi1'].iloc[:min_len].values**2 + df2['sigma_phi1'].iloc[:min_len].values**2) / theta_norm
    sigma_d_p1 = np.sqrt(df1['sigma_p1'].iloc[:min_len].values**2 + df2['sigma_p1'].iloc[:min_len].values**2) / p1_norm
    sigma_d_phi2 = np.sqrt(df1['sigma_phi2'].iloc[:min_len].values**2 + df2['sigma_phi2'].iloc[:min_len].values**2) / theta_norm
    sigma_d_p2 = np.sqrt(df1['sigma_p2'].iloc[:min_len].values**2 + df2['sigma_p2'].iloc[:min_len].values**2) / p2_norm

    sigma_delta = np.sqrt((d_phi1 * sigma_d_phi1)**2 + (d_p1 * sigma_d_p1)**2 + 
                          (d_phi2 * sigma_d_phi2)**2 + (d_p2 * sigma_d_p2)**2) / np.maximum(delta, 1e-6)

    delta_clean = np.maximum(delta, 1e-8)
    log_delta = np.log(delta_clean)

    # Fit over gekozen interval
    fit_mask = (t >= fit_start_s) & (t <= fit_end_s)
    t_fit = t[fit_mask]
    log_delta_fit = log_delta[fit_mask]

    poly, cov = np.polyfit(t_fit, log_delta_fit, 1, cov=True)
    lambda_val = poly[0]
    sigma_lambda = np.sqrt(cov[0, 0])
    intercept = poly[1]

    # --- PLOT 3: Delta + Fit + Foutmarge ---
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    axes[0].plot(t, delta, color='purple', label=r'$\delta(\tau)$')
    axes[0].fill_between(t, delta - sigma_delta, delta + sigma_delta, color='purple', alpha=0.2)
    axes[0].set_ylabel(r'$\delta(\tau)$ [lineair]')
    axes[0].set_title(f'Faseruimte Afstand met Onzekerheid - {label_pair}')
    axes[0].grid(True)

    axes[1].plot(t, log_delta, color='purple', label=r'$\ln \delta(\tau)$')
    axes[1].plot(t_fit, lambda_val * t_fit + intercept, 'r--', 
                 label=f'Fit: $\lambda = {lambda_val:.3f} \pm {sigma_lambda:.3f}$ s$^{{-1}}$')
    axes[1].axvspan(fit_start_s, fit_end_s, color='yellow', alpha=0.3, label='Fit Interval')
    axes[1].set_xlabel(r'Tijd $\tau$ [s]')
    axes[1].set_ylabel(r'$\ln \delta(\tau)$')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(output_dir / f"lyapunov_fit_{label_pair}.png")
    plt.close()

    return {
        "label_pair": label_pair,
        "lambda": lambda_val,
        "sigma_lambda": sigma_lambda,
        "fit_interval": f"[{fit_start_s}, {fit_end_s}]"
    }