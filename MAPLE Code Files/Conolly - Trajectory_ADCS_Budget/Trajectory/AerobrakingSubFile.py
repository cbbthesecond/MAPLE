#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from astropy import units as u
import jax
import jax.numpy as jnp
from diffrax import diffeqsolve, ODETerm, Dopri5, SaveAt
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# -----------------------------
# 1. GLOBAL CONSTANTS
# -----------------------------
R_MARS = 3389.5e3 * u.m
GM_MARS = 4.2828e13 * u.m**3 / u.s**2
Q_FACTOR = 1.85e-4  # placeholder

# -----------------------------
# 2. ORBITAL HELPER FUNCTIONS
# -----------------------------
def state_from_coe(a: float, e: float, i: float, RAAN: float, argp: float, nu: float, mu: float) -> np.ndarray:
    p = a*(1 - e**2)
    r = p/(1 + e*np.cos(nu))
    r_pf = np.array([r*np.cos(nu), r*np.sin(nu), 0.0])
    v_pf = np.array([
        -np.sqrt(mu/p)*np.sin(nu),
         np.sqrt(mu/p)*(e + np.cos(nu)),
         0.0
    ])
    R3_RAAN = np.array([
        [np.cos(RAAN), -np.sin(RAAN), 0],
        [np.sin(RAAN),  np.cos(RAAN), 0],
        [0,             0,            1]
    ])
    R1_i = np.array([
        [1, 0,           0],
        [0, np.cos(i),  -np.sin(i)],
        [0, np.sin(i),   np.cos(i)]
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

def coe_from_state(state: np.ndarray, mu: float) -> dict:
    r_vec = state[:3]
    v_vec = state[3:]
    r = np.linalg.norm(r_vec)
    v = np.linalg.norm(v_vec)
    E = 0.5*v**2 - mu/r
    if abs(E)>1e-14:
        a = -mu/(2*E)
    else:
        a = np.inf
    h_vec = np.cross(r_vec, v_vec)
    h = np.linalg.norm(h_vec)
    i = 0.0
    if h>1e-14:
        i = np.arccos(h_vec[2]/h)
    k_hat = np.array([0,0,1])
    n_vec = np.cross(k_hat, h_vec)
    n = np.linalg.norm(n_vec)
    RAAN = 0.0
    if n>1e-14:
        RAAN = np.arccos(np.clip(n_vec[0]/n, -1,1))
        if n_vec[1]<0:
            RAAN = 2*np.pi - RAAN
    e_vec = (1/mu)*((v**2 - mu/r)*r_vec - np.dot(r_vec,v_vec)*v_vec)
    e = np.linalg.norm(e_vec)
    argp = 0.0
    nu = 0.0
    if n>1e-14 and e>1e-14:
        argp = np.arctan2(np.dot(np.cross(n_vec, e_vec), h_vec)/(n*h),
                          np.dot(n_vec, e_vec)/n)
        nu = np.arctan2(np.dot(np.cross(e_vec, r_vec), h_vec)/(h*e),
                        np.dot(e_vec, r_vec)/e)
    elif e<=1e-14:
        nu = np.arccos(np.clip(np.dot(r_vec, v_vec)/(r*v), -1,1))
    if a==np.inf:
        r_peri = r
        r_apo = r
    else:
        r_peri = a*(1-e)
        r_apo  = a*(1+e)
    return {"a":a, "e":e, "i":i, "RAAN":RAAN, "argp":argp, "nu":nu,
            "r_peri":r_peri, "r_apo":r_apo}

# -----------------------------
# 3. ATMOSPHERE + DRAG
# -----------------------------
def mars_atmospheric_density_diffrax(altitude: float) -> float:
    def low_alt(_):
        T_C = -31.0 - 0.000998*altitude
        p_kPa = 0.699*jnp.exp(-0.00009*altitude)
        return T_C, p_kPa
    def high_alt(_):
        T_C = -23.4 - 0.00222*altitude
        p_kPa = 0.699*jnp.exp(-0.00009*altitude)
        return T_C, p_kPa
    T_C, p_kPa = jax.lax.cond(altitude<=7000.0, low_alt, high_alt, operand=None)
    T_K = jnp.maximum(T_C+273.15, 1.0)
    return p_kPa/(0.1921*T_K)

def adaptive_raise_periapsis(state: np.ndarray, desired_peri_alt_m: float, mu: float):
    coe_now = coe_from_state(state, mu)
    current_peri = coe_now["r_peri"]
    R_planet = R_MARS.value
    if current_peri >= (R_planet+desired_peri_alt_m):
        return state, 0.0
    delta_raise = 5000.0
    new_peri_alt = min(desired_peri_alt_m, current_peri-R_planet+delta_raise)
    new_r_peri = R_planet + new_peri_alt
    r_apo = coe_now["r_apo"]
    new_a = (r_apo+new_r_peri)/2.0
    new_e = (r_apo-new_r_peri)/(r_apo+new_r_peri)
    new_state = state_from_coe(new_a, new_e, coe_now["i"], coe_now["RAAN"], coe_now["argp"], 0.0, mu)
    old_v_peri = np.sqrt(mu*(1+coe_now["e"])/coe_now["r_peri"])
    new_v_peri = np.sqrt(mu*(1+new_e)/new_r_peri)
    dv = abs(new_v_peri-old_v_peri)
    return new_state, dv

def simulate_drag_pass_diffrax(state_entry: np.ndarray, mu: float, R_M: float, h_atm: float,
                               C_d: float, A: float, m: float, Q_factor: float,
                               desired_peri_alt_m: float, dt: float=1.0, t_max: float=10.0,
                               f_min_threshold: float=0.2, atmosphere_threshold: float=250e3,
                               phase: float=0.0):
    y_current = jnp.array(state_entry)
    t_total = 0.0
    integrated_heating = 0.0
    final_f = 1.0
    adaptive_dv = 0.0

    # PD Gains
    Kp = 18  
    Kd = 0.15
    old_error_deg = 0.0

    desired_incl_deg = 92.78
    final_bank_cmd_deg = 0.0

    while t_total < t_max:
        r_vec = y_current[:3]
        v_vec = y_current[3:]
        r = float(jnp.linalg.norm(r_vec))
        alt = r - R_M

        # compute inclination & error
        h_vec = jnp.cross(r_vec, v_vec)
        h_norm = jnp.linalg.norm(h_vec)+1e-12
        current_incl_rad = jnp.arccos(h_vec[2]/h_norm)
        current_incl_deg = current_incl_rad*180.0/jnp.pi

        error_deg = desired_incl_deg - current_incl_deg
        derror_deg = error_deg - old_error_deg
        old_error_deg = error_deg

        # PD
        bank_cmd_deg = -phase*(Kp*error_deg + Kd*derror_deg)
        bank_cmd_deg = jnp.clip(bank_cmd_deg, -95.0, 95.0)
        final_bank_cmd_deg = bank_cmd_deg  # store for return
        beta = bank_cmd_deg*(jnp.pi/180.0)

        if alt >= atmosphere_threshold:
            a_grav = -mu*r_vec/((r+1e-12)**3)
            v_new = v_vec+a_grav*dt
            r_new = r_vec+v_vec*dt
            y_current = jnp.concatenate([r_new, v_new])
        else:
            def drag_ode_pd(t, y, _args):
                rv = y[:3]
                vv = y[3:]
                rr = jnp.linalg.norm(rv)
                alt2 = rr - R_M
                rho = mars_atmospheric_density_diffrax(alt2)
                vmag = jnp.linalg.norm(vv)
                a_d = 0.5*rho*(vmag**2)*C_d*A/m
                v_hat = vv/(vmag+1e-12)
                h_hat = jnp.cross(rv,vv)/(jnp.linalg.norm(jnp.cross(rv,vv))+1e-12)
                a_drag = -a_d*(jnp.cos(beta)*v_hat + jnp.sin(beta)*h_hat)
                a_grav = -mu*rv/((rr+1e-12)**3)
                return jnp.concatenate([vv, a_grav+a_drag])

            sol = diffeqsolve(
                ODETerm(drag_ode_pd),
                Dopri5(),
                t0=t_total,
                t1=t_total+dt,
                dt0=0.1,
                y0=y_current,
                args=None,
                saveat=SaveAt(ts=[t_total+dt])
            )
            y_current = sol.ys[-1]

        t_total += dt

        # check peri => raise if needed
        coe_now = coe_from_state(np.array(y_current), mu)
        if coe_now["r_peri"] < (R_M + desired_peri_alt_m):
            from_state, dv_corr = adaptive_raise_periapsis(np.array(y_current), desired_peri_alt_m, mu)
            adaptive_dv += dv_corr
            y_current = jnp.array(from_state)

    # convert final bank to python float
    final_bank_cmd_deg_float = float(final_bank_cmd_deg)

    # Return the final state, plus the last commanded bank angle
    return np.array(y_current), final_bank_cmd_deg_float, integrated_heating, final_f, 0.0, adaptive_dv

# -----------------------------
# 4. MULTI-PASS AEROBRAKING
# -----------------------------
def multi_pass_aerobraking_fixed_peri(r_initial: u.Quantity, v_initial: u.Quantity,
                                      desired_peri_alt_m: float, target_apo_alt_km: float,
                                      mu: float, R_M: float, C_d=2.2, A=17.5967, m=2000.0,
                                      h_atm=200e3, Q_factor=1.85e-4, max_passes=1000,
                                      f_min_threshold=0.2):
    desired_r_peri = R_M + desired_peri_alt_m
    target_radius = R_M + target_apo_alt_km*1e3
    state_current = np.concatenate([r_initial.value, v_initial.value])
    coe = coe_from_state(state_current, mu)
    impulsive_dv_total = 0.0

    

    # If initial peri < desired, raise it
    if coe["r_peri"]<desired_r_peri:
        r_apo = coe["r_apo"]
        new_a = (r_apo+desired_r_peri)/2.0
        new_e = (r_apo-desired_r_peri)/(r_apo+desired_r_peri)
        state_current = state_from_coe(new_a, new_e, coe["i"], coe["RAAN"], coe["argp"], 0.0, mu)
        coe = coe_from_state(state_current, mu)
        

    original_apoapsis = coe["r_apo"]
    passes = 0
    total_time = 0.0
    total_heating = 0.0

    pass_data = []
    while coe["r_apo"]>target_radius and passes<max_passes:
        passes += 1
        progress = np.clip(1-(coe["r_apo"]-target_radius)/(original_apoapsis-target_radius), 0,1)
  
        st_exit, real_bank_angle_deg, Q_pass, final_f, _, adaptive_dv = simulate_drag_pass_diffrax(
            state_current, mu, R_M, h_atm, C_d, A, m, Q_factor,
            desired_peri_alt_m=desired_peri_alt_m,
            dt=1.0, t_max=10.0,
            f_min_threshold=f_min_threshold,
            atmosphere_threshold=250e3,
            phase=progress
        )
        state_current = st_exit
        coe_after = coe_from_state(state_current, mu)
        if coe_after["a"]<1e12:
            T_orbit = 2*np.pi*np.sqrt(coe_after["a"]**3/mu)
        else:
            T_orbit = 1e4
        total_time += T_orbit
        total_heating += Q_pass
        impulsive_dv_total += adaptive_dv
        coe = coe_after

        

        pass_data.append({
            "pass_number": passes,
            "peri_altitude_m": coe["r_peri"]-R_M,
            "impulsive_dv_m_s": adaptive_dv,
            "orbit_time_s": T_orbit,
            "inclination_rad": coe["i"],
            "inclination_deg": np.degrees(coe["i"]),
            "bank_angle_deg": real_bank_angle_deg,
            "final_state": state_current.copy()
        })

    if coe["r_apo"]>target_radius:
        new_a = (target_radius+desired_r_peri)/2.0
        new_e = (target_radius-desired_r_peri)/(target_radius+desired_r_peri)
        state_current = state_from_coe(new_a, new_e, coe["i"], coe["RAAN"], coe["argp"], 0.0, mu)
        coe = coe_from_state(state_current, mu)
        

    return coe, state_current, total_heating, passes, total_time, impulsive_dv_total, pass_data, original_apoapsis

# -----------------------------
# 5. FINAL CIRCULARIZATION
# -----------------------------
def final_circularization_burn(state: np.ndarray, mu: float) -> tuple[np.ndarray, float]:
    r_vec = state[:3]
    v_vec = state[3:]
    r = np.linalg.norm(r_vec)

    r_hat = r_vec / (r + 1e-12)
    h_vec = np.cross(r_vec, v_vec)
    h_norm = np.linalg.norm(h_vec) + 1e-12
    h_hat = h_vec / h_norm

    # Tangential direction t_hat in the same plane
    # cross(h_hat, r_hat) => a direction orthonormal to both
    t_hat = np.cross(h_hat, r_hat)
    t_norm = np.linalg.norm(t_hat) + 1e-12
    t_hat = t_hat / t_norm

    # 3) Circular speed at this radius
    v_circ = np.sqrt(mu / r)

    if np.dot(v_vec, t_hat) < 0.0:
        t_hat = -t_hat

    v_des = v_circ * t_hat

    # 5) Delta-v
    delta_v_vec = v_des - v_vec
    delta_v = np.linalg.norm(delta_v_vec)

    # 6) Build the new final state
    new_state = np.concatenate([r_vec, v_des])

    return new_state, delta_v

# -----------------------------
# 6. ORBIT PLOTTING (3D AND 2D)
# -----------------------------
def sample_orbit_in_3d(state: np.ndarray, mu: float, num_points: int=200):
    coe = coe_from_state(state, mu)
    a, e, inc, RAAN, argp = coe["a"], coe["e"], coe["i"], coe["RAAN"], coe["argp"]
    p = a*(1-e**2)
    nus = np.linspace(0,2*np.pi,num_points)

    R3_RAAN = np.array([
        [np.cos(RAAN), -np.sin(RAAN), 0],
        [np.sin(RAAN),  np.cos(RAAN), 0],
        [0,             0,            1]
    ])
    R1_i = np.array([
        [1, 0,           0],
        [0, np.cos(inc), -np.sin(inc)],
        [0, np.sin(inc),  np.cos(inc)]
    ])
    R3_argp = np.array([
        [np.cos(argp), -np.sin(argp), 0],
        [np.sin(argp),  np.cos(argp), 0],
        [0,             0,            1]
    ])
    Q = R3_RAAN @ R1_i @ R3_argp

    X, Y, Z = [], [], []
    for nu in nus:
        r = p/(1+e*np.cos(nu))
        r_pf = np.array([r*np.cos(nu), r*np.sin(nu), 0.0])
        r_3d = Q@r_pf
        X.append(r_3d[0])
        Y.append(r_3d[1])
        Z.append(r_3d[2])
    return np.array(X), np.array(Y), np.array(Z)

def plot_orbits_3d(pass_data, final_state_burn: np.ndarray, mu: float):
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("Mars Aerobraking Orbits (3D)")
    ax.set_box_aspect((1,1,1))

    # Mars sphere
    n_sphere = 50
    import math
    phi = np.linspace(0, math.pi, n_sphere)
    theta = np.linspace(0, 2*math.pi, n_sphere)
    phi, theta = np.meshgrid(phi, theta)
    r_mars = R_MARS.value
    Xs = r_mars*np.sin(phi)*np.cos(theta)
    Ys = r_mars*np.sin(phi)*np.sin(theta)
    Zs = r_mars*np.cos(phi)
    ax.plot_surface(Xs, Ys, Zs, color='orange', alpha=0.4, linewidth=0)

    passes_to_plot = [0, len(pass_data)-1]
    for i in range(0, len(pass_data), 50):
        passes_to_plot.append(i)
    passes_to_plot = sorted(list(set(passes_to_plot)))

    for idx in passes_to_plot:
        st = pass_data[idx]["final_state"]
        Xo, Yo, Zo = sample_orbit_in_3d(st, mu)
        if idx==0:
            label="Orbit #1"
        elif idx==(len(pass_data)-1):
            label="Last Orbit"
        else:
            label=f"Pass #{idx+1}"
        ax.plot(Xo, Yo, Zo, label=label)

    Xf, Yf, Zf = sample_orbit_in_3d(final_state_burn, mu)
    ax.plot(Xf, Yf, Zf, color='red', label="Final Circular Orbit", linewidth=2)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_aspect('equal', 'box')
    ax.legend(loc='best')
    plt.tight_layout()
    plt.savefig("orbits_around_mars_3d.png", dpi=300)
    plt.show()
    plt.close(fig)

def plot_orbits_2d(pass_data, final_state_burn: np.ndarray, mu: float):
    fig, ax = plt.subplots(figsize=(8,8))
    ax.set_title("Mars Aerobraking Orbits (2D top-down)")

    # Mars circle
    r_mars = R_MARS.value
    circle = plt.Circle((0,0), r_mars, color='orange', alpha=0.4)
    ax.add_patch(circle)

    passes_to_plot = [0, len(pass_data)-1]
    for i in range(0, len(pass_data), 50):
        passes_to_plot.append(i)
    passes_to_plot = sorted(list(set(passes_to_plot)))

    for idx in passes_to_plot:
        st = pass_data[idx]["final_state"]
        Xo, Yo, Zo = sample_orbit_in_3d(st, mu)
        if idx==0:
            label="Orbit #1"
        elif idx==(len(pass_data)-1):
            label="Last Orbit"
        else:
            label=f"Pass #{idx+1}"
        ax.plot(Xo, Zo, label=label)

    Xf, Yf, Zf = sample_orbit_in_3d(final_state_burn, mu)
    ax.plot(Xf, Zf, color='red', label="Final Circular Orbit", linewidth=2)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Z (m)")
    ax.set_aspect('equal', 'box')
    ax.legend(loc='best')
    plt.tight_layout()
    plt.savefig("orbits_around_mars_2d.png", dpi=300)
    plt.show()
    plt.close(fig)

# -----------------------------
# 7. PASS RESULTS PLOTTING
# -----------------------------
def plot_trajectory_analysis(pass_data):
    plt.style.use('ggplot')
    plt.rcParams.update({
        'figure.figsize': (10,6),
        'figure.dpi': 150,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'lines.linewidth': 2,
        'lines.markersize': 5,
    })

    passes = [d["pass_number"] for d in pass_data]
    bank_angles = [d["bank_angle_deg"] for d in pass_data]
    inclinations = [d["inclination_deg"] for d in pass_data]
    peri_altitudes = [d["peri_altitude_m"] for d in pass_data]
    impulsive_dv = [d["impulsive_dv_m_s"] for d in pass_data]

    # 1) Bank Angle + Inclination
    fig, ax1 = plt.subplots()
    line1 = ax1.plot(passes, bank_angles, color='blue', marker='o', label='Bank Angle (deg)', alpha=0.8)
    ax1.set_xlabel('Pass Number')
    ax1.set_ylabel('Bank Angle (deg)', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax2 = ax1.twinx()
    line2 = ax2.plot(passes, inclinations, color='red', marker='s', label='Inclination (deg)', alpha=0.8)
    ax2.set_ylabel('Inclination (deg)', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left')
    ax1.set_title('Bank Angle and Inclination vs Pass Number')
    fig.tight_layout()
    plt.savefig("bank_angle_vs_inclination.png", dpi=300)
    plt.close(fig)

    # 2) Periapsis Altitude vs Pass
    fig2, ax_alt = plt.subplots()
    ax_alt.plot(passes, peri_altitudes, color='green', marker='o', alpha=0.8, label='Periapsis Altitude (m)')
    ax_alt.set_xlabel('Pass Number')
    ax_alt.set_ylabel('Periapsis Altitude (m)', color='green')
    ax_alt.set_ylim(150000, 150050)
    ax_alt.tick_params(axis='y', labelcolor='green')
    ax_alt.set_title('Periapsis Altitude vs Pass Number')
    ax_alt.legend(loc='upper left')
    fig2.tight_layout()
    plt.savefig("periapsis_altitude_vs_passes.png", dpi=300)
    plt.close(fig2)

    # 3) Impulsive Δv vs Pass
    fig3, ax_dv = plt.subplots()
    ax_dv.plot(passes, impulsive_dv, color='magenta', marker='o', alpha=0.8, label='Impulsive Δv (m/s)')
    ax_dv.set_xlabel('Pass Number')
    ax_dv.set_ylabel('Impulsive Δv (m/s)', color='magenta')
    ax_dv.tick_params(axis='y', labelcolor='magenta')
    ax_dv.set_title('Impulsive Δv vs Pass Number')
    ax_dv.legend(loc='upper left')
    fig3.tight_layout()
    plt.savefig("impulsive_dv_vs_passes.png", dpi=300)
    plt.close(fig3)

# -----------------------------
# 8. "6-Month" LOGIC (PLACEHOLDER)
# -----------------------------
def simulate_campaign_with_initial_apoapsis(candidate_apo_alt_m: float) -> float:
    return 5.0e6  # ~58 days placeholder

def find_max_apoapsis_for_6_months(lower_bound_m: float, upper_bound_m: float, tol=1e3) -> float:
    target_time = 6*30.44*86400
    lb = lower_bound_m
    ub = upper_bound_m
    while (ub-lb)>tol:
        mid = (lb+ub)/2.0
        tof = simulate_campaign_with_initial_apoapsis(mid)
        if tof<=target_time:
            lb = mid
        else:
            ub = mid
    return (lb+ub)/2.0

# -----------------------------
# 9. MAIN
# -----------------------------
def main():
    # 1) Find candidate initial apo for 6-month
    lower_bound = 4.0e7
    upper_bound = 7.0e7
    candidate_initial_apo = find_max_apoapsis_for_6_months(lower_bound, upper_bound, tol=1e3)
    print(f"[Binary Search] Candidate initial apoapsis: {candidate_initial_apo:.1f} m")
    campaign_tof = simulate_campaign_with_initial_apoapsis(candidate_initial_apo)
    print(f"[Binary Search] Time-of-flight: {campaign_tof:.1f} s")

    # 2) Full aerobraking sim
    desired_peri_alt_m = 150000.0
    r_peri = R_MARS.value + desired_peri_alt_m
    r_apo = R_MARS.value + candidate_initial_apo
    a_init = (r_peri+r_apo)/2.0
    e_init = (r_apo-r_peri)/(r_apo+r_peri)

    # Enter at 90° => i=π/2
    i_init = np.pi/2
    RAAN_init=0.0
    argp_init=0.0
    nu_init=0.0

    st_init = state_from_coe(a_init, e_init, i_init, RAAN_init, argp_init, nu_init, GM_MARS.value)

    final_coe, final_st, total_heating, passes, tof_aero_sec, impulsive_dv_total, pass_data, orig_apo = \
        multi_pass_aerobraking_fixed_peri(
            r_initial=st_init[:3]*u.m,
            v_initial=st_init[3:]*u.m/u.s,
            desired_peri_alt_m=desired_peri_alt_m,
            target_apo_alt_km=350.0,
            mu=GM_MARS.value, R_M=R_MARS.value,
            C_d=1.0, A=17.5967, m=2000.0,
            h_atm=200e3, Q_factor=Q_FACTOR
        )

    print("\n=== Aerobraking Summary ===")
    print(f"Final # of passes: {passes}")
    print(f"Final aerobrake orbit COE: {final_coe}")
    print(f"Original capture apo: {orig_apo:.1f} m")
    print(f"Time-of-flight aerobraking: {tof_aero_sec/86400:.2f} days")
    print(f"Impulsive dv total (during passes): {impulsive_dv_total:.2f} m/s")

    # 3) Final circularization burn
    R_target = R_MARS.value + 380e3
    from_state_burn, dv_final = final_circularization_burn(final_st, GM_MARS.value)
    total_dv_usage = impulsive_dv_total + dv_final
    final_coe_after = coe_from_state(from_state_burn, GM_MARS.value)
    final_incl_deg = np.degrees(final_coe_after["i"])

    print("\n=== Final Circularization Burn ===")
    print(f"Delta-v for final burn: {dv_final:.3f} m/s")
    print(f"Final orbit COE: {final_coe_after}")
    print(f"Final inclination: {final_incl_deg:.2f} deg")
    print(f"Total dv usage (including final burn): {total_dv_usage:.2f} m/s")

    # 4) Pass-based plots
    plot_trajectory_analysis(pass_data)

    # 5) 3D orbits
    plot_orbits_3d(pass_data, from_state_burn, GM_MARS.value)

    # 6) 2D top-down orbits
    plot_orbits_2d(pass_data, from_state_burn, GM_MARS.value)

if __name__=="__main__":
    main()
