#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import spiceypy as spice
from astropy import units as u
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
import multiprocessing
from astropy import constants as const
from poliastro.iod import lambert

# --- Constants ---
AU_TO_KM = const.au.to(u.km).value
GM_SUN_VAL = const.GM_sun.value
GM_SUN = GM_SUN_VAL * u.m**3 / u.s**2

# --- Caching and SPICE Setup ---
cached_spice_state_cache = {}

def cached_spice_state(body_name, et, frame, observer_name, gm_sun_value=GM_SUN_VAL):
    cache_key = (body_name, et, frame, observer_name)
    if cache_key in cached_spice_state_cache:
        return cached_spice_state_cache[cache_key]
    else:
        state, light_time = spice.spkezr(body_name, et, frame, "LT+S", observer_name)
        cached_spice_state_cache[cache_key] = (state, light_time)
        return state, light_time

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
            try:
                spice.furnsh(kernel_path)
                print(f"Loaded kernel: {kernel_name} from {kernel_path}")
                total = spice.ktotal("ALL")
                print(f"Total kernels loaded after {kernel_name}: {total}")
            except spice.SpiceyError as e:
                print(f"Error loading kernel {kernel_name}: {e}")
                raise
        else:
            raise FileNotFoundError(f"SPICE kernel file not found: {kernel_name} at {kernel_path}")

    try:
        # Try to retrieve the leap seconds variable
        leap_seconds = spice.gcpool("DELTA_AT", 0, 100)
        if leap_seconds is not None:
            print("Leap seconds data in kernel pool:", leap_seconds)
        else:
            print("Warning: Leap seconds data not found in kernel pool.")
    except Exception as e:
        print("Error retrieving leap seconds data from kernel pool:", e)
        # Workaround: inject default leap seconds value
        default_leap_seconds = [37]
        spice.pdpool("DELTA_AT", default_leap_seconds)
        print("Manually loaded default DELTA_AT:", default_leap_seconds)
        try:
            leap_seconds = spice.gcpool("DELTA_AT", 0, 100)
            print("After injection, leap seconds data:", leap_seconds)
        except Exception as e2:
            print("Still unable to retrieve leap seconds data:", e2)
    total = spice.ktotal("ALL")
    print("Final total number of kernels loaded:", total)

def convert_datetime_to_et_spice(date_time):
    return spice.datetime2et(date_time)

def convert_et_to_datetime_utc(ephemeris_time):
    return spice.et2datetime(ephemeris_time).replace(tzinfo=None)

# --- Lambert / v∞ calculations ---
def calculate_v_infinity_spice(departure_date_et, time_of_flight_days,
                               target_body_name, departure_body_name,
                               gm_sun, AU_TO_KM):
    arrival_date_et = departure_date_et + time_of_flight_days * 86400.0

    earth_state, _ = cached_spice_state(departure_body_name, departure_date_et, "J2000", "SUN")
    earth_state = np.array(earth_state).flatten()
    earth_r = (earth_state[:3] * 1e3) * u.m
    earth_v = (earth_state[3:] * 1e3) * (u.m / u.s)

    mars_state, _ = cached_spice_state(target_body_name, arrival_date_et, "J2000", "SUN")
    mars_state = np.array(mars_state).flatten()
    mars_r = (mars_state[:3] * 1e3) * u.m
    mars_v = (mars_state[3:] * 1e3) * (u.m / u.s)

    try:
        v_dep, v_arr = lambert(gm_sun, earth_r, mars_r, (time_of_flight_days * u.day))
    except Exception as e:
        print(f"Lambert Solver Failed (poliastro): {e}")
        return np.nan * u.km/u.s, arrival_date_et, None

    v_infinity_vec = (v_arr - mars_v).to(u.km/u.s)

    mars_state_at_arrival, _ = cached_spice_state(target_body_name, arrival_date_et, "J2000", "SUN")
    mars_state_at_arrival = np.array(mars_state_at_arrival).flatten()
    mars_velocity_at_arrival = (mars_state_at_arrival[3:] * 1e3) * (u.m / u.s)

    return v_infinity_vec, arrival_date_et, mars_velocity_at_arrival

# --- Porkchop Calculation ---
def calculate_porkchop_point(params):
    dep_date_str, tof_days, departure_body_name, target_body_name, gm_sun_val, AU_TO_KM_val = params
    departure_datetime = datetime.strptime(dep_date_str, '%Y-%m-%d')
    departure_date_et = convert_datetime_to_et_spice(departure_datetime)
    v_inf_arr_vec, arrival_date_et, mars_velocity_at_arrival = calculate_v_infinity_spice(
        departure_date_et, tof_days, target_body_name, departure_body_name,
        gm_sun=GM_SUN, AU_TO_KM=AU_TO_KM_val)
    if (v_inf_arr_vec is not None) and (not np.any(np.isnan(v_inf_arr_vec.value))):
        v_inf_mag = np.linalg.norm(v_inf_arr_vec.value)
        earth_state, _ = cached_spice_state(departure_body_name, departure_date_et, "J2000", "SUN")
        earth_state = np.array(earth_state).flatten()
        earth_v_dep = (earth_state[3:] * 1e3) * (u.m / u.s)
        dot_val = np.dot(earth_v_dep[:2].value, v_inf_arr_vec[:2].value)
        norm1 = np.linalg.norm(earth_v_dep[:2].value)
        norm2 = np.linalg.norm(v_inf_arr_vec[:2].value)
        cos_angle = dot_val/(norm1*norm2) if (norm1>0 and norm2>0) else 1.0
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        prograde_angle = np.degrees(np.arccos(cos_angle))
        return dep_date_str, tof_days, v_inf_mag, float(prograde_angle)
    else:
        return dep_date_str, tof_days, np.nan, np.nan

def porkchop_plot_spice(departure_body_name, target_body_name,
                        search_start_str, search_end_str,
                        step_dep, min_tof, max_tof, step_tof,
                        contour_levels=20):
    departure_start_date = datetime.strptime(search_start_str, '%Y-%m-%d')
    departure_end_date   = datetime.strptime(search_end_str, '%Y-%m-%d')
    departure_dates = []
    current_date = departure_start_date
    while current_date <= departure_end_date:
        departure_dates.append(current_date.strftime('%Y-%m-%d'))
        current_date += timedelta(days=step_dep)

    time_of_flights = np.arange(min_tof, max_tof + step_tof, step_tof)
    v_infinity_data = np.zeros((len(departure_dates), len(time_of_flights)))
    arrival_angle_data = np.zeros((len(departure_dates), len(time_of_flights)))

    task_params = []
    for dep_date_str in departure_dates:
        for tof_days in time_of_flights:
            task_params.append((dep_date_str, tof_days,
                                departure_body_name, target_body_name,
                                GM_SUN_VAL, AU_TO_KM))

    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count(),
                                initializer=initialize_spice_kernels)
    results = pool.map(calculate_porkchop_point, task_params)
    pool.close()
    pool.join()

    for dep_date_str, tof_days, v_inf_val, angle_degrees in results:
        i = departure_dates.index(dep_date_str)
        j = np.where(time_of_flights == tof_days)[0][0]
        v_infinity_data[i, j]   = v_inf_val
        arrival_angle_data[i,j] = angle_degrees

    X, Y = np.meshgrid(time_of_flights, np.arange(len(departure_dates)))
    Z_vinf = v_infinity_data

    fig, ax = plt.subplots(figsize=(10,8))
    cset = ax.contourf(X, Y, Z_vinf, levels=contour_levels, cmap='viridis')
    cbar = fig.colorbar(cset, ax=ax)
    cbar.set_label('V_infinity (km/s)')
    ax.set_xlabel('Time of Flight (days)')
    ax.set_ylabel('Departure Date index')
    ax.set_title(f'Porkchop: {departure_body_name} to {target_body_name}')

    plt.tight_layout()
    plot_filename = f"porkchop_plots/porkchop_{departure_body_name}_to_{target_body_name}_{search_start_str}_{search_end_str}.png"
    if not os.path.exists("porkchop_plots"):
        os.makedirs("porkchop_plots")
    plt.savefig(plot_filename)
    plt.close(fig)

    return v_infinity_data, arrival_angle_data, departure_dates, time_of_flights

def compute_injection_burn(v_dep_lambert, earth_v_dep):
    v1 = v_dep_lambert.to(u.m/u.s)
    v2 = earth_v_dep.to(u.m/u.s)
    dv_injection_vector = v1 - v2
    dv_injection_mag    = np.linalg.norm(dv_injection_vector.value)
    return dv_injection_mag * u.m/u.s

def get_best_departure_info():
    departure_body_name_local = "EARTH"
    target_body_name_local    = "MARS"
    search_start_str_local    = "2028-01-01"
    search_end_str_local      = "2031-12-31"
    step_dep = 2
    step_tof = 2
    min_tof  = 150
    max_tof  = 540

    v_infinity_data_from_plot, arrival_angle_data_from_plot, \
        departure_dates_from_plot, time_of_flights_from_plot = porkchop_plot_spice(
            departure_body_name_local, target_body_name_local,
            search_start_str_local, search_end_str_local,
            step_dep, min_tof, max_tof, step_tof
        )

    min_v_inf = np.nanmin(v_infinity_data_from_plot)
    v_inf_tolerance = 0.1
    feasible_indices = np.where(v_infinity_data_from_plot <= (min_v_inf + v_inf_tolerance))

    if feasible_indices[0].size > 0:
        feasible_angles = arrival_angle_data_from_plot[feasible_indices]
        best_angle_index_within_feasible = np.nanargmin(feasible_angles)
        best_departure_date_index_angle  = feasible_indices[0][best_angle_index_within_feasible]
        best_tof_index_angle            = feasible_indices[1][best_angle_index_within_feasible]
        best_departure_date_str         = departure_dates_from_plot[best_departure_date_index_angle]
        best_departure_date_angle       = datetime.strptime(best_departure_date_str, '%Y-%m-%d')
        best_tof_angle                  = time_of_flights_from_plot[best_tof_index_angle]
        best_departure_date_et_angle    = convert_datetime_to_et_spice(best_departure_date_angle)

        best_v_inf_arr_angle, best_arrival_date_et_angle, best_mars_velocity_at_arrival_angle = calculate_v_infinity_spice(
            best_departure_date_et_angle, best_tof_angle,
            target_body_name_local, departure_body_name_local,
            gm_sun=GM_SUN, AU_TO_KM=AU_TO_KM
        )

        earth_state_angle, _ = cached_spice_state("EARTH", best_departure_date_et_angle, "J2000", "SUN")
        earth_state_angle = np.array(earth_state_angle).flatten()
        earth_v_angle = (earth_state_angle[3:] * 1e3) * (u.m/u.s)

        mars_state_angle, _ = cached_spice_state("MARS", best_arrival_date_et_angle, "J2000", "SUN")
        mars_state_angle = np.array(mars_state_angle).flatten()
        mars_r_angle = (mars_state_angle[:3] * 1e3) * u.m

        try:
            v_dep_angle, v_arr_angle = lambert(GM_SUN,
                                              (earth_state_angle[:3]*1e3)*u.m,
                                              mars_r_angle,
                                              best_tof_angle*u.day)
            dv_injection_angle = compute_injection_burn(v_dep_angle, earth_v_angle)
        except Exception as e:
            print(f"Warning: Lambert solver failed for prograde-optimized departure: {e}")
            dv_injection_angle = np.nan

        if not np.isnan(dv_injection_angle.value):
            print("\n=== BEST DEPARTURE INFORMATION (PROGRADE PRIORITY) ===")
            print("Choosing departure with best prograde angle within V-inf tolerance.")
            return {
                "best_departure_date": best_departure_date_angle,
                "best_arrival_date_et": best_arrival_date_et_angle,
                "dv_injection": dv_injection_angle,
                "best_tof": best_tof_angle,
                "v_inf_arr": best_v_inf_arr_angle,
                "best_mars_velocity_at_arrival": best_mars_velocity_at_arrival_angle,
                "total_dv": dv_injection_angle,
                "departure_time": best_departure_date_angle,
                "time_of_flight": best_tof_angle
            }

        print("Falling back to minimum V-infinity departure due to Lambert failure on prograde attempt.")

    best_v_inf_mag = np.nanmin(v_infinity_data_from_plot)
    best_indices   = np.where(np.nan_to_num(v_infinity_data_from_plot, nan=np.inf) == best_v_inf_mag)
    best_departure_date_index = best_indices[0][0]
    best_tof_index            = best_indices[1][0]
    best_departure_date_str   = departure_dates_from_plot[best_departure_date_index]
    best_departure_date       = datetime.strptime(best_departure_date_str, '%Y-%m-%d')
    best_tof                  = time_of_flights_from_plot[best_tof_index]
    best_departure_date_et    = convert_datetime_to_et_spice(best_departure_date)

    best_v_inf_arr, best_arrival_date_et, best_mars_velocity_at_arrival = calculate_v_infinity_spice(
        best_departure_date_et, best_tof, target_body_name_local, departure_body_name_local,
        gm_sun=GM_SUN, AU_TO_KM=AU_TO_KM
    )

    earth_state_min_vinf, _ = cached_spice_state("EARTH", best_departure_date_et, "J2000", "SUN")
    earth_state_min_vinf = np.array(earth_state_min_vinf).flatten()
    earth_v_min_vinf = (earth_state_min_vinf[3:] * 1e3) * (u.m / u.s)

    mars_state_min_vinf, _  = cached_spice_state("MARS", best_arrival_date_et, "J2000", "SUN")
    mars_state_min_vinf = np.array(mars_state_min_vinf).flatten()
    mars_r_min_vinf     = (mars_state_min_vinf[:3] * 1e3)*u.m

    try:
        v_dep_min_vinf, v_arr_min_vinf = lambert(GM_SUN,
                                                (earth_state_min_vinf[:3]*1e3)*u.m,
                                                mars_r_min_vinf,
                                                best_tof*u.day)
        dv_injection_min_vinf = compute_injection_burn(v_dep_min_vinf, earth_v_min_vinf)
    except Exception as e:
        print(f"Warning: Lambert solver failed for min-Vinf departure: {e}")
        dv_injection_min_vinf = np.nan*u.m/u.s

    return {
        "best_departure_date": best_departure_date,
        "best_arrival_date_et": best_arrival_date_et,
        "dv_injection": dv_injection_min_vinf,
        "best_tof": best_tof,
        "v_inf_arr": best_v_inf_arr,
        "best_mars_velocity_at_arrival": best_mars_velocity_at_arrival,
        "total_dv": dv_injection_min_vinf,
        "departure_time": best_departure_date,
        "time_of_flight": best_tof
    }

def main():
    initialize_spice_kernels()
    results = get_best_departure_info()

    best_dep   = results["best_departure_date"]
    best_arr_et= results["best_arrival_date_et"]
    dv_inj     = results["dv_injection"]
    best_tof   = results["best_tof"]
    v_inf_arr  = results["v_inf_arr"]
    mars_v_arr = results["best_mars_velocity_at_arrival"]

    total_dv   = results["total_dv"]
    dep_time   = results["departure_time"]
    tof_days   = results["time_of_flight"]

    print("\n=== BEST DEPARTURE INFORMATION ===")
    print(f"Departure Date: {best_dep.strftime('%Y-%m-%d')}")
    arrival_datetime_utc = convert_et_to_datetime_utc(best_arr_et)
    print(f"Arrival Date (UTC): {arrival_datetime_utc.strftime('%Y-%m-%d')}")
    print(f"Time of Flight: {best_tof:.2f} days")
    print(f"V-infinity at Mars Arrival: {np.linalg.norm(v_inf_arr.value):.3f} km/s")
    print(f"Injection Burn Δv: {dv_inj.to(u.m/u.s).value:.3f} m/s")
    print(f"TOTAL Δv (same as injection here): {total_dv.to(u.m/u.s).value:.3f} m/s")
    print("=====================================")

    print(f"\n(Also returning departure_time={dep_time}, time_of_flight={tof_days}, total_dv={total_dv})")

if __name__ == "__main__":
    results = main()
    with open("aerobraking_log.txt", "w", encoding="utf-8") as f:
        f.write(str(results))
