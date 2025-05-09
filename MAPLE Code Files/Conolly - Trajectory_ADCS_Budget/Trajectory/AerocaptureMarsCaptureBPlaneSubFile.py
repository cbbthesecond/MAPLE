#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import time
import numpy as np
from astropy import units as u
import astropy.constants as const
from scipy.integrate import solve_ivp
import spiceypy as spice

print("Process ID:", os.getpid())
# Ensure the current directory is in sys.path so that modules in the same folder can be imported.
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

try:
    from LaunchWindowOptimizationSubFile import get_best_departure_info
except Exception as e:
    print("WARNING: Could not import from LaunchWindowOptimizationSubFile. Using defaults.")
    print("Exception during import:", e)
    get_best_departure_info = None

# -----------------------------
# SPICE Kernel Initialization (for this module)
# -----------------------------
def initialize_spice_kernels():
    current_directory = os.path.dirname(os.path.abspath(__file__))
    kernel_directory = os.path.join(current_directory, 'spice_kernels')
    kernels_to_load = [
        "naif0012.tls",
        "de442.bsp",
        "mar097.bsp",
        "pck00011.tpc"
    ]
    for kernel_name in kernels_to_load:
        kernel_path = os.path.join(kernel_directory, kernel_name)
        if os.path.exists(kernel_path):
            spice.furnsh(kernel_path)
            print(f"Loaded kernel: {kernel_name} from {kernel_path}")
        else:
            raise FileNotFoundError(f"SPICE kernel file not found: {kernel_name} at {kernel_path}")
    total = spice.ktotal("ALL")
    print("Final total number of kernels loaded:", total)
    
    try:
        delta_at = spice.gcpool("DELTA_AT", 0, 100)
        print("Retrieved DELTA_AT:", delta_at)
    except Exception as e:
        print("Error retrieving DELTA_AT:", e)
        default_leap_seconds = [37]
        spice.pdpool("DELTA_AT", default_leap_seconds)
        print("Manually loaded default DELTA_AT:", default_leap_seconds)
        try:
            delta_at = spice.gcpool("DELTA_AT", 0, 100)
            print("After injection, DELTA_AT:", delta_at)
        except Exception as e2:
            print("Still unable to retrieve DELTA_AT:", e2)

# -----------------------------
# Basic Constants and Helpers
# -----------------------------
R_MARS = 3389.5e3 * u.m
GM_MARS = 4.2828e13 * u.m**3 / u.s**2
# Updated REF_AREA and SC_MASS for a larger spacecraft.
REF_AREA = 17.5967 * u.m**2       
SC_MASS = 2000 * u.kg         

def state_from_coe(a, e, i, RAAN, argp, nu, mu):
    p = a * (1 - e**2)
    r = p / (1 + e * np.cos(nu))
    r_pf = np.array([r * np.cos(nu), r * np.sin(nu), 0.0])
    v_pf = np.array([
        -np.sqrt(mu / p) * np.sin(nu),
         np.sqrt(mu / p) * (e + np.cos(nu)),
         0.0
    ])
    R3_RAAN = np.array([
        [np.cos(RAAN), -np.sin(RAAN), 0],
        [np.sin(RAAN),  np.cos(RAAN), 0],
        [0,             0,            1]
    ])
    R1_i = np.array([
        [1, 0,            0],
        [0, np.cos(i), -np.sin(i)],
        [0, np.sin(i),  np.cos(i)]
    ])
    R3_argp = np.array([
        [np.cos(argp), -np.sin(argp), 0],
        [np.sin(argp),  np.cos(argp), 0],
        [0,             0,            1]
    ])
    Q = R3_RAAN @ R1_i @ R3_argp
    r_vec = Q @ r_pf
    v_vec = Q @ v_pf
    return np.concatenate([r_vec, v_vec])

def get_drag_coefficient(mach_number):
    if mach_number < 0.8:
        return 1.5
    elif 0.8 <= mach_number < 1.2:
        return 1.8
    else:
        return 1.6

def mars_atmospheric_density(altitude):
    h = altitude.to(u.m).value
    if h <= 7000:
        T_celsius = -31.0 - 0.000998 * h
        p_kPa = 0.699 * np.exp(-0.00009 * h)
    else:
        T_celsius = -23.4 - 0.00222 * h
        p_kPa = 0.699 * np.exp(-0.00009 * h)
    T_val = T_celsius + 273.15
    if T_val < 1.0:
        T_val = 1.0
    p_val = p_kPa * 1e3
    rho_float = p_val / (0.1921 * T_val)
    return rho_float * (u.kg / u.m**3)

def mars_speed_of_sound(altitude):
    h = altitude.to(u.m).value
    if h <= 7000:
        T_celsius = -31 - 0.000998 * h
    else:
        T_celsius = -23.4 - 0.00222 * h
    T_val = max(0.1, T_celsius + 273.15)
    gamma_co2 = 1.3
    R_spec_co2 = 188.92
    c = np.sqrt(gamma_co2 * R_spec_co2 * T_val)
    return c * (u.m / u.s)

# -----------------------------
# Thermal Loading Functions
# -----------------------------
# Simple convective heating model:
#     q_dot = K_HEATING * sqrt(rho) * v_p^3
# Threshold set to 150 kW/m^2.
K_HEATING = 1.83  # [W/(m^2)]/( (kg/m^3)^0.5*(m/s)^3 )
THERMAL_THRESHOLD = 1.5e6  # W/m^2

def compute_heat_flux(altitude, v_inf, mu, k_heating):
    r = R_MARS.to(u.m).value + altitude
    v_p = np.sqrt(v_inf**2 + 2 * mu / r)
    rho = mars_atmospheric_density(altitude * u.m).value
    q_dot = k_heating * np.sqrt(rho) * v_p**3
    return q_dot, v_p

def find_safe_periapsis_alt(v_inf, mu, threshold, k_heating, alt_min=100e3, alt_max=500e3, tol=1e2):
    low = alt_min
    high = alt_max
    safe_alt = high
    while high - low > tol:
        mid = (low + high) / 2.0
        q_dot, _ = compute_heat_flux(mid, v_inf, mu, k_heating)
        if q_dot > threshold*1e3:
            low = mid
        else:
            safe_alt = mid
            high = mid
    return safe_alt

# -----------------------------
# Drag Integration for Aerocapture
# -----------------------------
def simulate_aerocapture_drag(v_inf, r_p, mu, Cd, A, m, h_entry):
    e = 1 + (r_p * v_inf**2) / mu
    p = r_p * (1 + e)
    r_entry = R_MARS.to(u.m).value + h_entry
    cos_nu = (p / r_entry - 1) / e
    cos_nu = np.clip(cos_nu, -1.0, 1.0)
    nu_entry = -np.arccos(cos_nu)
    r0 = r_entry * np.array([np.cos(nu_entry), np.sin(nu_entry)])
    v0 = np.array([
        -np.sqrt(mu/p) * np.sin(nu_entry),
         np.sqrt(mu/p) * (e + np.cos(nu_entry))
    ])
    y0 = np.concatenate([r0, v0])
    
    # Initialize maximum drag force.
    F_drag_max = 0.0
    
    def ode(t, y):
        r_vec = y[0:2]
        v_vec = y[2:4]
        r_norm = np.linalg.norm(r_vec)
        alt = r_norm - R_MARS.to(u.m).value
        a_grav = -mu * r_vec / (r_norm**3)
        rho = mars_atmospheric_density(alt * u.m).value
        v_norm = np.linalg.norm(v_vec)
        if v_norm < 1e-6:
            a_drag = np.array([0.0, 0.0])
            F_drag = 0.0
        else:
            a_drag = -0.5 * Cd * A / m * rho * v_norm**2 * (v_vec / v_norm)
            # Drag force: F_drag = 0.5 * Cd * A * rho * v^2 * normalization factor.
            F_drag = 0.5 * Cd * A * rho * v_norm**2 *0.443
        nonlocal F_drag_max
        if F_drag > F_drag_max:
            F_drag_max = F_drag
        return np.concatenate([v_vec, a_grav + a_drag])
    
    def event_exit(t, y):
        r_vec = y[0:2]
        r_norm = np.linalg.norm(r_vec)
        alt = r_norm - R_MARS.to(u.m).value
        return alt - h_entry
    event_exit.terminal = True
    event_exit.direction = 1
    
    t_span = (0, 100)
    sol = solve_ivp(ode, t_span, y0, events=event_exit, max_step=0.1)
    y_final = sol.y[:, -1]
    v_exit = np.linalg.norm(y_final[2:4])
    return v_exit, F_drag_max

# -----------------------------
# Modified Single-Pass Aerocapture Calculation
# -----------------------------
def single_pass_aerocapture_dv(v_inf_arr, apo_target_m, include_drag=True, Cd=1.8,
                               A=REF_AREA.to(u.m**2).value, m=SC_MASS.to(u.kg).value,
                               h_entry=250e3):
    v_inf_m_s = v_inf_arr.to(u.m / u.s).value
    v_mag = np.linalg.norm(v_inf_m_s)
    if v_mag > 6000.0:
        print("v_inf too large for feasible aerocapture.")
        return 1e9, None, None, None, "v_inf_too_high", None

    # Adjust periapsis altitude based on thermal loading.
    safe_peri_alt = find_safe_periapsis_alt(v_mag, GM_MARS.value, THERMAL_THRESHOLD, K_HEATING,
                                              alt_min=100e3, alt_max=500e3, tol=1e2)
    print(f"\n[Thermal Constraint] Selected periapsis altitude to limit heating: {safe_peri_alt:.0f} m above surface")
    q_dot, v_p = compute_heat_flux(safe_peri_alt, v_mag, GM_MARS.value, K_HEATING)
    print(f"[Thermal Loading] At periapsis altitude {safe_peri_alt:.0f} m: q_dot = {q_dot/1e9:.1f} MW/m², v_p = {v_p:.1f} m/s")
    
    R_p_capture = R_MARS + safe_peri_alt * u.m
    r_a_target = R_MARS + apo_target_m * u.m
    a_capture = (R_p_capture + r_a_target) / 2.0
    e_capture = (r_a_target - R_p_capture) / (r_a_target + R_p_capture)
    
    # Compute the target orbital speed at periapsis for the capture orbit.
    v_p_target = np.sqrt(GM_MARS.value * (2 / R_p_capture.value - 1 / a_capture.value))
    
    if include_drag:
        # Compute hyperbolic periapsis speed if there were no drag.
        v_p_init = np.sqrt(v_mag**2 + 2 * GM_MARS.value / R_p_capture.value)
        # Use drag integration to compute the effective speed at the exit of the atmosphere.
        r_p = R_p_capture.value
        v_effective, F_drag_max = simulate_aerocapture_drag(v_mag, r_p, GM_MARS.value, Cd, A, m, h_entry)
        print(f"\n[Drag Integration] v_exit after atmospheric pass: {v_effective:.1f} m/s")
        print(f"[Drag Integration] Maximum drag force encountered: {F_drag_max:.1f} N")
        # Calculate the reduction in velocity due to drag.
        drag_dv_reduction = v_p_init - v_effective
        print(f"[Drag Integration] Δv reduction from drag: {drag_dv_reduction:.1f} m/s")
    else:
        v_effective = np.sqrt(v_mag**2 + 2 * GM_MARS.value / R_p_capture.value)
    
    dv_capture = v_effective - v_p_target
    if dv_capture < 0.0:
        dv_capture = 0.0

    print("\n=== Single-Pass Aerocapture Summary ===")
    print(f"Selected periapsis altitude: {R_p_capture.value - R_MARS.value:.2f} m")
    print(f"Capture orbit: a = {a_capture.value:.1f} m, e = {e_capture:.3f}, target apoapsis = {r_a_target.value - R_MARS.value:.1f} m")
    if include_drag:
        print(f"v_exit (after drag integration) = {v_effective:.1f} m/s")
    else:
        print(f"v_p_initial (no drag) = {v_effective:.1f} m/s")
    print(f"v_p_target = {v_p_target:.1f} m/s")
    print(f"Capture Δv = {dv_capture:.1f} m/s")

    capture_state_apo = state_from_coe(a_capture.value, e_capture, 0.0, 0.0, 0.0, np.pi, GM_MARS.value)
    method = "single_pass_drag"
    return dv_capture, (R_p_capture - R_MARS).to(u.m).value, (r_a_target - R_MARS).to(u.m).value, e_capture, method, capture_state_apo

def get_mars_capture_dv(v_inf_arr, apo_target_m=66249542.2, include_drag=True):
    dv_capture, periapsis_alt_m, target_apoapsis_m, e_capture, method, capture_state_apo = single_pass_aerocapture_dv(
        v_inf_arr, apo_target_m, include_drag=include_drag)
    return {
        "dv_capture": dv_capture,
        "periapsis_alt_m": periapsis_alt_m,
        "apoapsis_alt_m": target_apoapsis_m,
        "ecc": e_capture,
        "method": method,
        "capture_state_apo": capture_state_apo,
    }

# -----------------------------
# MAIN EXECUTION WITH EXPLICIT B‑PLANE TARGETING
# -----------------------------
def main():
    initialize_spice_kernels()
    results = get_best_departure_info() if get_best_departure_info is not None else None

    if results is None:
        print("WARNING: Could not import from LaunchWindowOptimizationSubFile. Using defaults.\n")
        dv_injection = 0.0 * u.m / u.s
        best_tof = 999.0
        # Default v_inf (in km/s) converted to astropy Quantity
        v_inf_arr = np.array([4.5, 0, 0]) * (u.km / u.s)
    else:
        dv_injection = results["dv_injection"]
        best_tof = results["best_tof"]
        v_inf_arr = results["v_inf_arr"]
        print("\n=== BEST DEPARTURE INFORMATION (PROGRADE PRIORITY) ===")
        print("Using data from LaunchWindowOptimizationSubFile.")

    # -----------------------------
    # B‑Plane Targeting Correction
    # -----------------------------
    # Define target periapsis altitude for aerocapture (here, base value of 170 km above surface used in B-plane logic)
    R_p_final = R_MARS + 170e3 * u.m

    # Convert incoming hyperbolic excess velocity to m/s
    v_inf_vec = v_inf_arr.to(u.m/u.s).value
    v_inf_mag = np.linalg.norm(v_inf_vec)

    # --- Define B-plane Coordinate Axes ---
    S_hat = v_inf_vec / v_inf_mag
    ref_axis = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(S_hat, ref_axis)) > 0.9:
        ref_axis = np.array([1.0, 0.0, 0.0])
    T_hat = np.cross(ref_axis, S_hat)
    T_hat = T_hat / np.linalg.norm(T_hat)
    R_hat = np.cross(S_hat, T_hat)
    R_hat = R_hat / np.linalg.norm(R_hat)

    # --- Compute Desired B‑Plane Intercept ---
    mu_mars = GM_MARS.value
    R_p_final_val = R_p_final.to(u.m).value
    e_hyper = 1 + (R_p_final_val * v_inf_mag**2) / mu_mars
    a_hyper = - mu_mars / (v_inf_mag**2)
    B_mag = abs(a_hyper) * np.sqrt(e_hyper**2 - 1)
    B_dot_target   = B_mag
    B_cross_target = 0.0
    print("\n=== B‑PLANE TARGETING PARAMETERS ===")
    print(f"Target periapsis altitude: {(R_p_final - R_MARS).to(u.km):.1f} km")
    print(f"Desired B‑plane intercept: B_dot = {B_dot_target/1000:.1f} km, B_cross = {B_cross_target/1000:.1f} km")

    # --- Estimate Current B‑Plane Intercept (Pre‑Correction) ---
    safe_alt = 300e3 * u.m
    R_p_current = R_MARS + safe_alt
    R_p_current_val = R_p_current.to(u.m).value
    e_hyper_cur = 1 + (R_p_current_val * v_inf_mag**2) / mu_mars
    a_hyper_cur = - mu_mars / (v_inf_mag**2)
    B_cur_mag  = abs(a_hyper_cur) * np.sqrt(e_hyper_cur**2 - 1)
    B_dot_current   = B_cur_mag
    B_cross_current = 0.0

    # --- Determine the Required Change in B‑Plane Coordinates ---
    dB_dot   = B_dot_target - B_dot_current
    dB_cross = B_cross_target - B_cross_current

    # --- Compute the Correction ΔV ---
    dv_inplane = np.zeros(3)
    if abs(dB_dot) > 0:
        theta_inplane = dB_dot / B_cur_mag
        dv_inplane_mag = v_inf_mag * abs(theta_inplane)
        dv_inplane_dir = T_hat if dB_dot > 0 else -T_hat
        dv_inplane = dv_inplane_mag * dv_inplane_dir

    dv_outplane = np.zeros(3)
    if abs(dB_cross) > 0:
        theta_outplane = dB_cross / B_cur_mag
        dv_outplane_mag = v_inf_mag * abs(theta_outplane)
        dv_outplane_dir = R_hat if dB_cross > 0 else -R_hat
        dv_outplane = dv_outplane_mag * dv_outplane_dir

    dv_correction_vec = dv_inplane + dv_outplane
    dv_correction = np.linalg.norm(dv_correction_vec)

    # Update the arrival velocity vector with the correction burn
    v_inf_corrected_vec = v_inf_vec + dv_correction_vec
    v_inf_corrected = v_inf_corrected_vec * (u.m/u.s)

    print("\n=== B‑PLANE TARGETING CORRECTION ===")
    print(f"Pre-correction periapsis (assumed): {(R_p_current - R_MARS).to(u.km):.1f} km above surface")
    print(f"Pre-correction B‑plane intercept: B_dot = {B_dot_current/1000:.1f} km, B_cross = {B_cross_current/1000:.1f} km")
    print(f"Required B‑plane shift: ΔB_dot = {dB_dot/1000:.1f} km, ΔB_cross = {dB_cross/1000:.1f} km")
    print(f"ΔV for B‑plane correction: {dv_correction:.1f} m/s")
    
    print(f"\nArrival V∞ (after B‑plane correction): {np.linalg.norm(v_inf_corrected.to(u.km/u.s).value):.3f} km/s")
    # Continue with aerocapture using the corrected v_inf, with integrated drag and thermal adjustment.
    capture_results = get_mars_capture_dv(v_inf_corrected, apo_target_m=66249542.2, include_drag=True)
    dv_capture = capture_results["dv_capture"]
    R_p = capture_results["periapsis_alt_m"]
    r_a = capture_results["apoapsis_alt_m"]
    e_target = capture_results["ecc"]
    method = capture_results["method"]
    capture_state_apo = capture_results["capture_state_apo"]

    print("\n=== AEROCAPTURE RESULTS (with B‑plane targeting) ===")
    print(f"Capture Δv: {dv_capture:.1f} m/s")
    print(f"Periapsis altitude: {(R_p*u.m + R_MARS - R_MARS).to(u.m):.1f} m above surface")
    print(f"Apoapsis altitude (target): {(r_a*u.m + R_MARS - R_MARS).to(u.m):.1f} m above surface")
    print(f"Eccentricity: {e_target:.2f}")
    print(f"Method used: {method}")
    print("Capture state at apoapsis:", capture_state_apo)
    
    total_dv = (dv_injection.to(u.m/u.s).value + dv_correction + dv_capture)
    print(f"Injection Δv: {dv_injection.to(u.m/u.s).value:.1f} m/s")
    print(f"TOTAL Δv (injection + correction + capture) = {total_dv:.1f} m/s")
    
    return {"dv_capture": dv_capture,
            "periapsis_alt_m": R_p,
            "apoapsis_alt_m": r_a,
            "ecc": e_target,
            "method": method,
            "capture_state_apo": capture_state_apo,
            "dv_injection": dv_injection.to(u.m/u.s).value,
            "DV_correction": dv_correction,
            "total_dv": total_dv}

if __name__ == "__main__":
    results = main()
    with open("aerobraking_log.txt", "w", encoding="utf-8") as f:
        f.write(str(results))
