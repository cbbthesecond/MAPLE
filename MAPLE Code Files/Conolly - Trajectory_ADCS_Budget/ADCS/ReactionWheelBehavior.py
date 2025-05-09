import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_continuous_are

# Quaternion utilities

def normalize_quaternion(q):
    return q / np.linalg.norm(q)

def quat_mult(q, r):
    q0, q1, q2, q3 = q
    r0, r1, r2, r3 = r
    return np.array([
        q0*r0 - q1*r1 - q2*r2 - q3*r3,
        q0*r1 + q1*r0 + q2*r3 - q3*r2,
        q0*r2 - q1*r3 + q2*r0 + q3*r1,
        q0*r3 + q1*r2 - q2*r1 + q3*r0
    ])

def quat_conjugate(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])

def quaternion_error_no_flip(q_cur, q_des):
    q_err = quat_mult(quat_conjugate(q_des), q_cur)
    w = np.clip(q_err[0], -1, 1)
    ang = 2.0 * np.arccos(w)
    signed = ang - 2*np.pi if ang > np.pi else ang
    axis = q_err[1:]
    norm = np.linalg.norm(axis)
    return np.zeros(3) if norm < 1e-12 else (axis / norm) * signed

def quat_from_angle_axis(deg, axis):
    axis = axis / np.linalg.norm(axis)
    half = np.radians(deg) / 2
    return np.array([np.cos(half), *(axis * np.sin(half))])

# Dynamics and RK4 integrator

def dynamics(state, t, params, ctrl=None, K_lqr=None):
    q = normalize_quaternion(state[:4])
    omega = state[4:7]
    wheels = state[7:11]
    e_int = state[11:14]
    e_rot = quaternion_error_no_flip(q, params['q_des'])

    if ctrl == 'LQR':
        tau_cmd = -K_lqr @ np.hstack((e_rot, omega))
    else:
        tau_cmd = -(params['Kp'] * e_rot + params['Ki'] * e_int + params['Kd'] * omega)

    tau_w = np.clip(np.linalg.pinv(params['S']) @ tau_cmd, -params['max_tau'], params['max_tau'])
    friction = params['b_visc'] * wheels + params['b_coul'] * np.sign(wheels)
    wheels_dot = (tau_w - friction) / params['J_w']
    tau_dist = params.get('tau_disturb', np.zeros(3))
    omega_dot = np.linalg.inv(params['I_s']) @ (params['S'] @ tau_w + tau_dist)

    dq = 0.5 * quat_mult(q, np.concatenate(([0.], omega)))
    e_int_dot = e_rot

    # Instantaneous mechanical power and positive-only consumption
    mech_power = np.dot(tau_w, wheels)
    power_consumed = max(mech_power, 0.0) / params['eta']
    E_dot = power_consumed

    return np.concatenate([dq, omega_dot, wheels_dot, e_int_dot, [E_dot]])

def rk4(state0, t, params, ctrl=None, K_lqr=None):
    dt = t[1] - t[0]
    ys = np.zeros((len(t), len(state0)))
    state = state0.copy()
    w_max = params['h_max'] / params['J_w']
    for i, ti in enumerate(t):
        ys[i] = state
        k1 = dynamics(state, ti, params, ctrl, K_lqr)
        k2 = dynamics(state + 0.5*dt*k1, ti + 0.5*dt, params, ctrl, K_lqr)
        k3 = dynamics(state + 0.5*dt*k2, ti + 0.5*dt, params, ctrl, K_lqr)
        k4 = dynamics(state + dt*k3, ti + dt, params, ctrl, K_lqr)
        state += dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        state[:4] = normalize_quaternion(state[:4])
        state[7:11] = np.clip(state[7:11], -w_max, w_max)
    return ys

# LQR design
def design_lqr(params):
    I = params['I_s']
    A = np.block([[np.zeros((3,3)), np.eye(3)], [np.zeros((3,3)), np.zeros((3,3))]])
    B = np.vstack([np.zeros((3,3)), np.linalg.inv(I)])
    P = solve_continuous_are(A, B, params['Q'], params['R'])
    return np.linalg.inv(params['R']) @ B.T @ P

# Performance metrics
def performance_metrics(t, err, init):
    tol = abs(init) * 0.02
    settling = np.nan
    for i in range(len(err)):
        if np.all(np.abs(err[i:]) <= tol):
            settling = t[i]
            break
    peak = np.max(err) if init < 0 else np.min(err)
    overshoot = abs(peak) / abs(init) * 100
    return settling, overshoot

if __name__ == '__main__':
    params = {'I_s': np.diag([8.,10.,12.]), 'J_w': 0.0127,
              'S': np.column_stack(([1,0,0],[0,1,0],[0,0,1],[1,1,1]) / np.sqrt(3)),
              'b_visc':1e-3, 'b_coul':1e-4, 'eta':0.9, 'h_max':8.0, 'max_tau':0.25,
              'Kp':2.0, 'Ki':0.2, 'Kd':5.0, 'q_des':np.array([1.,0.,0.,0.]),
              'Q':np.diag([500,500,500,5000,5000,5000]), 'R':np.eye(3)*0.5}

    q_init = quat_from_angle_axis(30, np.array([0,1,0]))
    if q_init[0] < 0: q_init = -q_init
    state0 = np.concatenate([q_init, np.zeros(3), np.zeros(4), np.zeros(3), [0.]])
    K_lqr = design_lqr(params)
    t = np.linspace(0,100,6000)
    ys_pid = rk4(state0, t, params, 'PID')
    ys_lqr = rk4(state0, t, params, 'LQR', K_lqr)

    def compute_signed_err(ys):
        errs = []
        for r in ys:
            q_err = quaternion_error_no_flip(normalize_quaternion(r[:4]), params['q_des'])
            errs.append(np.degrees(np.linalg.norm(q_err) * np.sign(q_err[1])))
        return np.array(errs)

    err_pid = compute_signed_err(ys_pid)
    err_lqr = compute_signed_err(ys_lqr)
    sp_pid, os_pid = performance_metrics(t, err_pid, 30)
    sp_lqr, os_lqr = performance_metrics(t, err_lqr, 30)
    print(f"PID Settling Time: {sp_pid:.2f}s, Overshoot: {os_pid:.2f}%")
    print(f"LQR Settling Time: {sp_lqr:.2f}s, Overshoot: {os_lqr:.2f}%")

    # Gravity-gradient disturbance test
    mu = 4.282837e13
    R_mars = 3390e3
    alt = 400e3
    r = R_mars + alt
    I_vals = np.diag(params['I_s'])
    tau_gg = (3*mu/(r**3))*(np.max(I_vals)-np.min(I_vals))/2
    params['tau_disturb'] = np.array([0., tau_gg, 0.])
    state0_hold = np.concatenate([params['q_des'], np.zeros(3), np.zeros(4), np.zeros(3), [0.]])
    T_orbit = 2*np.pi * np.sqrt(r**3/mu)
    t2 = np.linspace(0, T_orbit, 1000)
    ys_hold = rk4(state0_hold, t2, params, 'PID')
    energy_per_orbit = ys_hold[-1, -1]
    orbits_per_year = (365*24*3600) / T_orbit
    power_per_orbit = energy_per_orbit / T_orbit
    power_per_year = energy_per_orbit * orbits_per_year / (365*24*3600)
    print(f"Gravity Gradient Torque: {tau_gg:.3e} N·m")
    print(f"Energy per orbit: {energy_per_orbit:.2f} J, avg power per orbit: {power_per_orbit:.4f} W")
    print(f"Estimated avg power per year: {power_per_year:.4f} W")
    fig, axs = plt.subplots(2,2, figsize=(12,10))
    axs[0,0].plot(t, err_pid, label='PID'); axs[0,0].plot(t, err_lqr, '--', label='LQR')
    axs[0,0].set_title('Signed Attitude Error vs Time'); axs[0,0].set_xlabel('Time [s]'); axs[0,0].set_ylabel('Error [deg]'); axs[0,0].legend()
    axs[0,1].plot(t, ys_pid[:,4], label='wx PID'); axs[0,1].plot(t, ys_pid[:,5], label='wy PID'); axs[0,1].plot(t, ys_pid[:,6], label='wz PID')
    axs[0,1].plot(t, ys_lqr[:,4], '--', label='wx LQR'); axs[0,1].plot(t, ys_lqr[:,5], '--', label='wy LQR'); axs[0,1].plot(t, ys_lqr[:,6], '--', label='wz LQR')
    axs[0,1].set_title('Body Angular Rates vs Time'); axs[0,1].set_xlabel('Time [s]'); axs[0,1].set_ylabel('Angular Rate [rad/s]'); axs[0,1].legend()
    for i, label in enumerate(['W1','W2','W3','W4']): axs[1,0].plot(t, ys_pid[:,7+i], label=f'{label} PID'); axs[1,0].plot(t, ys_lqr[:,7+i], '--', label=f'{label} LQR')
    axs[1,0].set_title('Reaction Wheel Speeds vs Time'); axs[1,0].set_xlabel('Time [s]'); axs[1,0].set_ylabel('Wheel Speed [rad/s]'); axs[1,0].legend(loc='lower right')
    axs[1,1].plot(t, ys_pid[:,-1], label='PID'); axs[1,1].plot(t, ys_lqr[:,-1], '--', label='LQR')
    axs[1,1].set_title('Cumulative Energy Usage vs Time'); axs[1,1].set_xlabel('Time [s]'); axs[1,1].set_ylabel('Energy [J]'); axs[1,1].legend()
    plt.tight_layout(rect=[0,0,1,0.96])
    plt.show()
