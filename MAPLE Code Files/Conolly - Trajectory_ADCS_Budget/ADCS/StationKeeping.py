#!/usr/bin/env python3
"""
Enhanced station-keeping ΔV model for a 350 km Mars orbit.
This script includes:
  - An improved atmospheric model with solar flux modulation.
  - Eclipse effects in the solar radiation pressure (SRP) calculation.
  - A rough estimation of ΔV required for J2 perturbation corrections.
"""

import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------
# Constants and Mars Parameters
# -------------------------------------
MU_MARS = 4.2828e13         # Mars gravitational parameter [m^3/s^2]
RADIUS_MARS = 3390e3        # Mars radius [m]
A_MARS = 1.524              # Mars semi-major axis [AU]
P0_SRP = 4.56e-6            # Solar radiation pressure at 1 AU [N/m^2]

# -------------------------------------
# Spacecraft Parameters for Drag/SRP
# -------------------------------------
SPACECRAFT = {
    "mass": 2000,          # spacecraft mass [kg]
    "area_drag": 17.575,   # effective cross-sectional area for drag [m^2]
    "C_D": 2.2,            # drag coefficient (typical value)
    "area_srp": 10,        # effective area for SRP [m^2]
    "reflectivity": 0.8    # reflectivity coefficient (0 for absorber, 1 for perfect reflector)
}

# -------------------------------------
# Mars Atmospheric Model Parameters
# -------------------------------------
# Two–regime exponential model:
#   - Below h_transition: lower atmosphere.
#   - Above h_transition: exospheric region with continued exponential decay.
ATMOSPHERE = {
    "h_transition": 80e3,    # Transition altitude [m]
    "p0": 610,               # Surface pressure [Pa]
    "T_lower": 210,          # Temperature in the lower atmosphere [K]
    "H_lower": 11000,        # Scale height for the lower atmosphere [m]
    "T_exo": 150,            # Exospheric temperature [K]
    "H_exo": 15000,          # Scale height for the exospheric region [m]
    "solar_flux_factor": 1.0 # Modulation factor (1.0 nominal, >1 increases density)
}

# -------------------------------------
# Improved Atmospheric Model
# -------------------------------------
def mars_atmosphere(h: float, solar_flux_factor: float = 1.0):
    """
    Compute Mars atmospheric properties at altitude h (m).
    Returns:
      T (K) : Temperature.
      p (Pa): Pressure.
      rho (kg/m^3): Density.
      
    The model applies a solar flux modulation factor to the density.
    """
    if h < 0:
        T = ATMOSPHERE["T_lower"]
        p = ATMOSPHERE["p0"]
    elif h < ATMOSPHERE["h_transition"]:
        T = ATMOSPHERE["T_lower"]
        p = ATMOSPHERE["p0"] * np.exp(-h / ATMOSPHERE["H_lower"])
    else:
        p_transition = ATMOSPHERE["p0"] * np.exp(-ATMOSPHERE["h_transition"] / ATMOSPHERE["H_lower"])
        T = ATMOSPHERE["T_exo"]
        p = p_transition * np.exp(-(h - ATMOSPHERE["h_transition"]) / ATMOSPHERE["H_exo"])
    
    # Ideal gas law for CO2-dominated Martian atmosphere
    R_specific = 189.0  # J/(kg*K) for CO2
    rho = p / (R_specific * T)
    # Apply solar activity modulation to the density
    rho *= solar_flux_factor
    return T, p, rho

# -------------------------------------
# Eclipse & Sunlight Fraction for SRP
# -------------------------------------
def sunlight_fraction(r: float):
    """
    Calculate the fraction of an orbit in sunlight.
    For a circular orbit, the eclipse half-angle is given by:
        theta = arcsin(RADIUS_MARS / r)
    and the eclipse fraction is theta/π.
    
    Returns the fraction of the orbit in sunlight.
    """
    if r <= RADIUS_MARS:
        return 0.0
    theta = np.arcsin(RADIUS_MARS / r)
    eclipse_fraction = theta / np.pi
    return 1 - eclipse_fraction

# -------------------------------------
# Orbital Parameters
# -------------------------------------
def orbital_parameters(altitude: float):
    """
    Compute orbital parameters for a circular orbit around Mars.
    Returns:
      r (m): Orbital radius.
      v (m/s): Orbital velocity.
      T_orbit (s): Orbital period.
    """
    r = RADIUS_MARS + altitude
    v = np.sqrt(MU_MARS / r)
    T_orbit = 2 * np.pi * np.sqrt(r**3 / MU_MARS)
    return r, v, T_orbit

# -------------------------------------
# ΔV Estimation: Atmospheric Drag
# -------------------------------------
def compute_drag_dv(altitude: float, T_orbit: float, v: float, spacecraft: dict,
                      solar_flux_factor: float = 1.0):
    """
    Calculate the annual ΔV due to atmospheric drag.
    """
    _, _, density = mars_atmosphere(altitude, solar_flux_factor)
    a_drag = 0.5 * density * v**2 * (spacecraft["C_D"] * spacecraft["area_drag"] / spacecraft["mass"])
    dv_drag_per_orbit = a_drag * T_orbit
    seconds_per_year = 365.25 * 24 * 3600
    orbits_per_year = seconds_per_year / T_orbit
    annual_dv_drag = dv_drag_per_orbit * orbits_per_year
    return annual_dv_drag, orbits_per_year

# -------------------------------------
# ΔV Estimation: Solar Radiation Pressure (SRP)
# -------------------------------------
def compute_srp_dv(T_orbit: float, spacecraft: dict, r: float):
    """
    Calculate the annual ΔV due to solar radiation pressure (SRP).
    This function accounts for:
      - Scaling of SRP with Mars’ distance.
      - The effect of reflectivity.
      - Eclipse (shadowing) reducing the effective SRP.
      - A geometry factor (0.1) to account for average incidence.
    """
    pressure_srp = P0_SRP / (A_MARS**2)
    effective_pressure = pressure_srp * (1 + spacecraft["reflectivity"])
    a_srp = effective_pressure * (spacecraft["area_srp"] / spacecraft["mass"])
    sun_frac = sunlight_fraction(r)
    # Apply a factor to account for incidence angles and average effectiveness
    a_srp_effective = a_srp * sun_frac * 0.1
    dv_srp_per_orbit = a_srp_effective * T_orbit
    seconds_per_year = 365.25 * 24 * 3600
    orbits_per_year = seconds_per_year / T_orbit
    annual_dv_srp = dv_srp_per_orbit * orbits_per_year
    return annual_dv_srp

# -------------------------------------
# ΔV Estimation: J2 Perturbations
# -------------------------------------
def compute_j2_dv(r: float, v: float):
    """
    Provide a rough estimate of the annual ΔV required to counteract perturbations
    due to Mars' J2 (oblateness). This is a simplified estimate:
    
      Δv ≈ correction_factor * J2 * (RADIUS_MARS/r)^2 * v
      
    The correction_factor is tunable based on detailed mission analysis.
    """
    J2 = 1.96045e-3  # Mars' J2 coefficient
    correction_factor = 0.3
    annual_dv_j2 = correction_factor * J2 * (RADIUS_MARS / r)**2 * v
    return annual_dv_j2

# -------------------------------------
# Main Routine
# -------------------------------------
def main():
    altitude = 380e3         # Orbit altitude [m]
    solar_flux_factor = 1.0  # Modify this value to simulate higher/lower solar activity
    
    # Compute basic orbital parameters
    r, v, T_orbit = orbital_parameters(altitude)
    
    # Retrieve atmospheric properties
    T_atm, p_atm, density = mars_atmosphere(altitude, solar_flux_factor)
    
    # Compute ΔV components
    annual_dv_drag, orbits_per_year = compute_drag_dv(altitude, T_orbit, v, SPACECRAFT, solar_flux_factor)
    annual_dv_srp = compute_srp_dv(T_orbit, SPACECRAFT, r)
    annual_dv_j2 = compute_j2_dv(r, v)
    
    # Total annual ΔV required for station keeping
    total_annual_dv = annual_dv_drag + annual_dv_srp + annual_dv_j2
    
    # Print computed parameters and ΔV budget
    print("Station-Keeping ΔV estimates for a 380 km Mars orbit:")
    print(f"Orbital radius: {r:,.2f} m")
    print(f"Orbital velocity: {v:,.2f} m/s")
    print(f"Orbital period: {T_orbit:,.2f} s")
    print(f"Orbits per year: {orbits_per_year:,.0f}")
    print(f"Atmospheric density at {altitude/1e3:.1f} km: {density:.3e} kg/m^3")
    print(f"Annual ΔV due to atmospheric drag: {annual_dv_drag:.3f} m/s")
    print(f"Annual ΔV offset from SRP (accounting for eclipse): {annual_dv_srp:.3f} m/s")
    print(f"Annual ΔV for J2 corrections: {annual_dv_j2:.3f} m/s")
    print(f"Total annual ΔV required: {total_annual_dv:.3f} m/s")
    
    # Plot cumulative ΔV over mission duration
    mission_years = 10
    years = np.arange(0, mission_years + 1)
    cumulative_drag = annual_dv_drag * years
    cumulative_srp = annual_dv_srp * years
    cumulative_j2 = annual_dv_j2 * years
    cumulative_total = cumulative_drag + cumulative_srp + cumulative_j2

    plt.figure(figsize=(8, 5))
    plt.plot(years, cumulative_drag, '--', label="Drag ΔV")
    plt.plot(years, cumulative_srp, '--', label="SRP ΔV")
    plt.plot(years, cumulative_j2, '--', label="J2 ΔV")
    plt.plot(years, cumulative_total, 'k-', linewidth=2, label="Total Cumulative ΔV")
    plt.xlabel("Mission Duration (years)")
    plt.ylabel("Cumulative ΔV (m/s)")
    plt.title("Enhanced Cumulative Station-Keeping ΔV for 380 km Mars Orbit")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
