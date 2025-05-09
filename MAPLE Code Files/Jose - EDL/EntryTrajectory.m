mu_Mars = 4.282837e13;           
R_Mars  = 3389.5e3; 
m       = 330;
Cd      = 1.5;     
A       = pi * (0.8)^2;     
beta    = m/(Cd*A);
rho0    = 0.02;
H       = 10000;
h    = 125000;
v    = 4500;
gamma= deg2rad(-12);
dt   = 0.1;
t    = 0;
time = []; 
alt = []; 
vel = []; 
Fpa = []; 
accel = [];
while true
    rho = rho0 * exp(-h / H);
    g   = mu_Mars / (R_Mars + h)^2;
    D   = 0.5 * rho * v^2 * Cd * A;
    dv_dt     = -D/m - g * sin(gamma);
    dgamma_dt = (v/(R_Mars + h) - g/v) * cos(gamma);
    dh_dt     = v * sin(gamma);
    v     = v + dv_dt * dt;
    gamma = gamma + dgamma_dt * dt;
    h     = h + dh_dt * dt;
    t     = t + dt;
    time(end+1) = t; 
    alt(end+1) = h; 
    vel(end+1) = v;
    accel(end+1) = -dv_dt; 
    Fpa(end+1)   = gamma;
    if v <= 400 || h <= 0
        fprintf('Parachute deploy triggered at t = %.1f s, alt = %.0f m, v = %.1f m/s\n', t, h, v);
        break;
    end
end

peak_accel = max(accel);
fprintf('Peak deceleration = %.1f m/s^2 (%.2f G)\n', peak_accel, peak_accel/9.81);

D_parachute = 16;
A_parachute = pi * (D_parachute/2)^2;
Cd_parachute= 0.7;
CdA_parachute= Cd_parachute * A_parachute;
h_chute = h;
v_chute = v;
t_chute = t;
h  = h_chute;
v  = v_chute;
t0 = t_chute;
dt = 0.1;
while h > 1000
    rho = rho0 * exp(-h / H);
    g   = mu_Mars / (R_Mars + h)^2;
    D   = 0.5 * rho * v^2 * CdA_parachute;
    dv_dt = g - D/m;
    v = v + dv_dt * dt;
    h = h - v * dt;
    t0 = t0 + dt;
    if mod(t0, 1) < dt
        fprintf('t=%.0f s: alt=%.0f m, v=%.1f m/s\n', t0, h, v);
    end
end
fprintf('Parachute descent: reached %.0f m altitude at %.1f m/s\n', h, v);

