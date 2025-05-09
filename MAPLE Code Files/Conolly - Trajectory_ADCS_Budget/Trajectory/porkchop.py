import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import datetime

import astrodynamics as ad
import lambert as lb

# global constants
mu = 1.32712428e11    # gravitational parameter of the sun (km^3/s^2)

# calculate c3 and delv values from lambert solution
def _get_lambert_estimates_(v_dep, v_arr, r1, r2, flight_time_secs,
                           orb_type, M, path):
    v1_list, v2_list = lb.solve(mu, r1, r2, flight_time_secs, orb_type, path, M)
    v1, v2 = v1_list[0], v2_list[0]
    # compute v_inf for departure and arrival (subtract planet velocities)
    v_inf_dep = np.linalg.norm(v_dep - v1) 
    v_inf_arr = np.linalg.norm(v_arr - v2)
    # characteristic energy
    c3_dep = v_inf_dep**2
    # total ΔV = v_inf_dep + v_inf_arr
    delv_total = v_inf_dep + v_inf_arr    
    return [c3_dep, delv_total]

# get plot data for a departure-arrival combination
def _get_porkchop_plot_data_(dep_planet_name, jd_dep, arr_planet_name, jd_arr):
    dep_planet_id = ad.get_planet_id(dep_planet_name)
    arr_planet_id = ad.get_planet_id(arr_planet_name)    
    coe_dep, r_dep, v_dep, jd_d = ad.get_planet_state_vector(mu, dep_planet_id, jd_dep)
    coe_arr, r_arr, v_arr, jd_a = ad.get_planet_state_vector(mu, arr_planet_id, jd_arr)
    jd_dep_str, jd_arr_str = ad.jd_str(jd_dep), ad.jd_str(jd_arr)
    flight_time_days = (jd_arr - jd_dep)
    flight_time_secs = flight_time_days * (24.0 * 60.0 * 60.0)
    
    # Lambert estimation for two different trajectory types (low and high path)
    orb_type, M, low_path = 'prograde', 0.0, 'low'
    c3_and_delv_1 = _get_lambert_estimates_(v_dep, v_arr, r_dep, r_arr,
                                             flight_time_secs, orb_type, M, low_path)
    
    orb_type, M, low_path = 'prograde', 0.0, 'high'
    c3_and_delv_2 = _get_lambert_estimates_(v_dep, v_arr, r_dep, r_arr,
                                             flight_time_secs, orb_type, M, low_path)
    out = jd_dep_str, jd_arr_str, flight_time_days, c3_and_delv_1, c3_and_delv_2, coe_dep, coe_arr
    return out

# Generate porkchop plot contour data over a grid of departure and arrival dates.
# Expanded here by allowing the caller to specify the span (in days) for departure and arrival.
def _generate_date_matrix_(jd_dep, jd_arr, dep_range=180, arr_range=400, dt_dep=2, dt_arr=5):
    # Check if the provided date range yields a valid time-of-flight (minimum of 60 days here)
    if (jd_arr - (jd_dep + dep_range)) < 60:
        diff = round(jd_arr - (jd_dep + dep_range), 2)
        raise Exception("error: tof is %s days. Change dep/arr dates." % diff)
    jd_dep_list = np.array(list(range(int(jd_dep), int(jd_dep + dep_range), int(dt_dep))))
    jd_arr_list = np.array(list(range(int(jd_arr), int(jd_arr + arr_range), int(dt_arr))))
    return jd_dep_list, jd_arr_list

# get porkchop plot contour data
def _generate_porkchop_plot_data_(dep_planet_name, jd_dep_list, arr_planet_name, jd_arr_list):
    jd_dep_str_list = np.empty(jd_dep_list.shape, dtype=object)
    jd_arr_str_list = np.empty(jd_arr_list.shape, dtype=object)
    contour_shape = (jd_arr_list.shape[0], jd_dep_list.shape[0])
    tof_days_list = np.zeros(contour_shape, dtype=np.float64)
    c3_dep_1_list = np.zeros(contour_shape, dtype=np.float64)
    c3_dep_2_list = np.zeros(contour_shape, dtype=np.float64)
    delv_t_1_list = np.zeros(contour_shape, dtype=np.float64)
    delv_t_2_list = np.zeros(contour_shape, dtype=np.float64)
    
    rows, cols = tof_days_list.shape
    for ix in range(rows):
        jd_arr_i = jd_arr_list[ix]
        for iy in range(cols):
            jd_dep_i = jd_dep_list[iy]
            res = _get_porkchop_plot_data_(dep_planet_name, jd_dep_i, arr_planet_name, jd_arr_i)
            jd_dep_str, jd_arr_str, flight_time_days, c3_and_delv_1, c3_and_delv_2, coe_dep, coe_arr = res
            c3_dep_1, delv_1_total = c3_and_delv_1
            c3_dep_2, delv_2_total = c3_and_delv_2
            tof_days_list[ix][iy] = flight_time_days
            c3_dep_1_list[ix][iy] = c3_dep_1
            c3_dep_2_list[ix][iy] = c3_dep_2
            delv_t_1_list[ix][iy] = delv_1_total
            delv_t_2_list[ix][iy] = delv_2_total
            jd_arr_str_list[ix] = jd_arr_str
            jd_dep_str_list[iy] = jd_dep_str
    out = jd_dep_str_list, jd_arr_str_list, tof_days_list, c3_dep_1_list, c3_dep_2_list, delv_t_1_list, delv_t_2_list
    return out

def plot_porkchop(title, xlist, ylist,
                  xy_contour_data_1, xy_contour_data_2, clevels,
                  xy_tof_data, tlevels):
    
    def set_ticks(ax):
        x_tick_spacing, y_tick_spacing = 5, 3
        ax.xaxis.set_major_locator(ticker.MultipleLocator(x_tick_spacing))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(y_tick_spacing))
        ax.xaxis.set_tick_params(labelsize=7, rotation=90)
        ax.yaxis.set_tick_params(labelsize=7)
        ax.set_xlabel("Dep Date (dd-mm-yyyy)", fontsize=8)
        ax.set_ylabel("Arr Date (dd-mm-yyyy)", fontsize=8)
        ax.grid(which='major', linestyle='dashdot', linewidth='0.5', color='gray')
        ax.grid(which='minor', linestyle='dotted', linewidth='0.5', color='gray')
        ax.minorticks_on()
        ax.tick_params(which='both', top=False, left=False, right=False, bottom=False)
        return

    def tp_fmt(x):
        return f"{x:.1f} days"
    
    plt.figure(figsize=(12,9))
    cp1 = plt.contour(xlist, ylist, xy_contour_data_1, clevels, cmap="rainbow")
    plt.clabel(cp1, inline=True, fontsize=7)
    cp2 = plt.contour(xlist, ylist, xy_contour_data_2, clevels, cmap="rainbow")
    plt.clabel(cp2, inline=True, fontsize=7)
    tp = plt.contour(xlist, ylist, xy_tof_data, tlevels, colors='k', linestyles=':')
    plt.clabel(tp, inline=True, fmt=tp_fmt, fontsize=7)
    plt.title(title, fontdict={'fontsize':8})
    set_ticks(plt.gca())
    plt.gca().set_aspect(1)
    plt.tight_layout()
    plt.show()
    return

# Modified main function to allow passing a wider range for the porkchop plot.
def make_porkchop_plot(dep_planet_name, dep_date, arr_planet_name, arr_date, plot_type='delv_plot',
                       dep_range_days=180, arr_range_days=400):
    try:
        dt_dep = datetime.datetime.strptime(dep_date, '%d-%m-%Y')
        dt_arr = datetime.datetime.strptime(arr_date, '%d-%m-%Y')
    except ValueError:
        print('error: wrong date format. Verify as dd-mm-yyyy')
        return
    
    d, m, y = dt_dep.day, dt_dep.month, dt_dep.year    
    jd_dep = ad.greg_to_jd(y, m, d)
    d, m, y = dt_arr.day, dt_arr.month, dt_arr.year     
    jd_arr = ad.greg_to_jd(y, m, d)
    
    # Generate date arrays with expanded ranges
    jd_dep_list, jd_arr_list = _generate_date_matrix_(jd_dep, jd_arr, dep_range=dep_range_days, arr_range=arr_range_days)
    
    res = _generate_porkchop_plot_data_(dep_planet_name, jd_dep_list, arr_planet_name, jd_arr_list)   
    jd_dep_str_list, jd_arr_str_list, tof_days_list, c3_dep_1_list, c3_dep_2_list, delv_t_1_list, delv_t_2_list = res
    
    c3_levels = [4, 5, 6, 8, 10, 12, 14, 16, 18, 19, 20, 30, 50, 70, 100,
                 150, 200, 250, 300, 350, 400, 450, 500,]
    t_levels  = [100, 150, 200, 250, 300, 350, 400, 450, 500, 550]
    
    if plot_type == 'delv_plot':
        title = ('Porkchop plot (ΔV Total = ||ΔV_' + dep_planet_name + '|| + ||ΔV_' + 
                 arr_planet_name + '||)\n')
        plot_porkchop(title, jd_dep_str_list, jd_arr_str_list,
                      c3_dep_1_list, c3_dep_2_list, c3_levels,
                      tof_days_list, t_levels)
    elif plot_type == 'c3_plot':       
        title = 'Porkchop plot (C3-characteristic energy = $v_{∞}^2$)\n'
        plot_porkchop(title, jd_dep_str_list, jd_arr_str_list,
                      delv_t_1_list, delv_t_2_list, c3_levels,
                      tof_days_list, t_levels)
    else:
        raise Exception("error: plot types should be 'c3_plot' or 'delv_plot'")
    return 

# main function
if __name__ == "__main__":
    dep_planet = 'Earth'
    dep_date = '05-01-2028'  # dd-mm-yyyy format
    arr_planet = 'Mars'
    arr_date = '01-01-2031'  # dd-mm-yyyy format
    
    # Choose the plot type ('delv_plot' or 'c3_plot')
    plot_values = 'delv_plot'
    # Expand the range: for example, use a full year (365 days) for departures 
    # and a wider arrival window (800 days) for more extended time-of-flight solutions.
    dep_range_days = 365
    arr_range_days = 800
    make_porkchop_plot(dep_planet, dep_date, arr_planet, arr_date, plot_values, dep_range_days, arr_range_days)
