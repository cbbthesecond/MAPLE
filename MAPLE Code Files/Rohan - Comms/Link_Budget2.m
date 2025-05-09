%% Mars Orbiter Communication System: Complete Link Budget Analysis with Power Calculations


clear; clc; close all;

%% Constants
c = 3e8;                % Speed of light [m/s]
k = 1.38e-23;           % Boltzmann constant [J/K]

%% Mission Parameters
d_mars_earth = 2.25e11; % Mars-Earth average distance [m]
d_surface_orbiter = 1000e3; % Surface-to-orbiter distance [m]

%% Frequencies
f_x = 8.41e9;           % X-band frequency [Hz]
f_ka = 32e9;            % Ka-band frequency [Hz]
f_uhf = 401.5e6;        % UHF relay frequency [Hz]

%% Transmitter Powers [W]
P_tx_x_W = 100;
P_tx_ka_W = 80;
P_tx_surface_W = 15;

%% Antenna Gains [dBi]
G_tx_x = 42; G_rx_x = 68;
G_tx_ka = 45; G_rx_ka = 70;
G_tx_surface = 2; G_rx_orbiter = 8;

%% Loss Factors [dB]
L_misc_ds = 3;
L_misc_surf = 2.5;

%% FSPL Calculation Function
fspl_dB = @(d,f) 20*log10((4*pi*d*f)/c);

% Calculate FSPL at nominal distances
L_fs_x_dB   = fspl_dB(d_mars_earth, f_x);
L_fs_ka_dB  = fspl_dB(d_mars_earth, f_ka);
L_fs_uhf_dB = fspl_dB(d_surface_orbiter, f_uhf);

%% Received Power Calculation Function
rx_power_dBm = @(Ptx,Gtx,Lfs,Lmisc,Grx) ...
    (10*log10(Ptx*1e3) + Gtx - Lfs - Lmisc + Grx);

% Calculate Received Powers
P_rx_x_dBm   = rx_power_dBm(P_tx_x_W, G_tx_x, L_fs_x_dB, L_misc_ds, G_rx_x);
P_rx_ka_dBm  = rx_power_dBm(P_tx_ka_W, G_tx_ka, L_fs_ka_dB, L_misc_ds, G_rx_ka);
P_rx_uhf_dBm = rx_power_dBm(P_tx_surface_W,G_tx_surface,L_fs_uhf_dB,L_misc_surf,G_rx_orbiter);

%% Noise Power Calculation Function
noise_power_dBm = @(Tsys,B) 10*log10(k*Tsys*B*1e3);

% Noise Parameters
T_sys_ds   = 500; B_ds   = 1e6;
T_sys_uhf  = 300; B_uhf  = 4e6;

% Calculate Noise Powers
P_noise_ds_dBm   = noise_power_dBm(T_sys_ds,B_ds);
P_noise_uhf_dBm  = noise_power_dBm(T_sys_uhf,B_uhf);

%% SNR Calculation
SNR_x_dB   = P_rx_x_dBm - P_noise_ds_dBm;
SNR_ka_dB  = P_rx_ka_dBm - P_noise_ds_dBm;
SNR_uhf_dB = P_rx_uhf_dBm - P_noise_uhf_dBm;

%% Channel Capacity Calculation Function (Shannon)
capacity_bps=@(B,SNRdB) B.*log2(1+10.^(SNRdB/10));

C_x_bps   = capacity_bps(B_ds,SNR_x_dB);
C_ka_bps  = capacity_bps(B_ds,SNR_ka_dB);
C_UHF_bps= capacity_bps(B_uhf,SNR_uhf_dB);

%% Effective Throughput (Assuming overhead)
overhead_ds=0.20; overhead_UHF=0.15;
throughput_x_bps   = C_x_bps*(1-overhead_ds);
throughput_ka_bps  = C_ka_bps*(1-overhead_ds);
throughput_UHF_bps= C_UHF_bps*(1-overhead_UHF);

%% Component-by-Component Power Calculations

% Define components and their peak/average power usage (in Watts)
components = {
    'X-band TWTA',          250,    62.5;   % Peak, Average (25% duty cycle)
    'Ka-band TWTA',         200,    50;     % Peak, Average (25% duty cycle)
    'UHF Transceiver',       28,    11.2;   % Peak, Average (40% duty cycle)
    'Antenna Pointing',      25,     2.5;   % Peak, Average (10% duty cycle)
    'RAD5545 SBC',           35,    35;     % Peak and Average (continuous operation)
    'Solid State Recorder',  15,    15;     % Peak and Average (continuous operation)
};

% Extract power data for calculations
component_names   = components(:,1);
peak_powers       = cell2mat(components(:,2));
average_powers    = cell2mat(components(:,3));

% Total power calculations
total_peak_power   = sum(peak_powers);
total_average_power= sum(average_powers);

%% Display Component-by-Component Power Data in Command Window
fprintf('\n--- Component-by-Component Power Usage ---\n');
fprintf('%-25s %-10s %-10s\n', 'Component', 'Peak Power (W)', 'Average Power (W)');
for i=1:length(component_names)
    fprintf('%-25s %-10.2f %-10.2f\n', component_names{i}, peak_powers(i), average_powers(i));
end
fprintf('------------------------------------------\n');
fprintf('Total Peak Power: %.2f W\n', total_peak_power);
fprintf('Total Average Power: %.2f W\n\n', total_average_power);

%% FSPL Line Graph vs Distance (Integrated)
d_min=5.5e10; d_max=4.0e11;
distance=linspace(d_min,d_max,1000);

FSPL_X_band=fspl_dB(distance,f_x);
FSPL_Ka_band=fspl_dB(distance,f_ka);

figure('Name','Free Space Path Loss vs Mars-Earth Distance');
plot(distance/1e9,FSPL_X_band,'b-','LineWidth',2); hold on;
plot(distance/1e9,FSPL_Ka_band,'r--','LineWidth',2); grid on;
xlabel('Mars-Earth Distance (Million km)');
ylabel('Free Space Path Loss (dB)');
title('FSPL vs Distance for Mars Orbiter Communication');
legend('X-band (8.41 GHz)','Ka-band (32 GHz)','Location','best');

surface_distance_km=linspace(200,4000,1000)*1e3;
FSPL_UHF_surface=fspl_dB(surface_distance_km,f_uhf);

figure('Name','UHF FSPL Surface-to-Orbiter');
plot(surface_distance_km/1e3,FSPL_UHF_surface,'g-','LineWidth',2); grid on;
xlabel('Surface-to-Orbiter Distance (km)');
ylabel('Free Space Path Loss (dB)');
title('UHF FSPL vs Surface-to-Orbiter Distance');
legend('UHF Relay Link');

%% Display Overall Link Budget Results Clearly
fprintf('\n--- Link Budget Summary ---\n');
fprintf('%-12s %-20s %-12s %-12s\n', 'Link', 'Received Power(dBm)', 'SNR(dB)', 'Throughput(Mbps)');
fprintf('%-12s %-20.2f %-12.2f %-12.2f\n', 'X-band', P_rx_x_dBm, SNR_x_dB, throughput_x_bps/1e6);
fprintf('%-12s %-20.2f %-12.2f %-12.2f\n', 'Ka-band', P_rx_ka_dBm, SNR_ka_dB, throughput_ka_bps/1e6);
fprintf('%-12s %-20.2f %-12.2f %-12.2f\n', 'UHF Relay', P_rx_uhf_dBm, SNR_uhf_dB, throughput_UHF_bps/1e6);



%% Constants
c = 3e8;                % Speed of light [m/s]
k = 1.38e-23;           % Boltzmann constant [J/K]

%% Mission Parameters
d_mars_earth = 2.25e11; % Mars-Earth average distance [m]
d_surface_orbiter = 1000e3; % Surface-to-orbiter distance [m]

%% Frequencies
f_x = 8.41e9;           % X-band frequency [Hz]
f_ka = 32e9;            % Ka-band frequency [Hz]
f_uhf = 401.5e6;        % UHF relay frequency [Hz]

%% Transmitter Powers [W]
P_tx_x_W = 100;
P_tx_ka_W = 80;
P_tx_surface_W = 15;

%% Antenna Gains [dBi]
G_tx_x = 42; G_rx_x = 68;
G_tx_ka = 45; G_rx_ka = 70;
G_tx_surface = 2; G_rx_orbiter = 8;

%% Loss Factors [dB]
L_misc_ds = 3;
L_misc_surf = 2.5;

%% FSPL Calculation Function
fspl_dB = @(d,f) 20*log10((4*pi*d*f)/c);

% Calculate FSPL at nominal distances
L_fs_x_dB   = fspl_dB(d_mars_earth, f_x);
L_fs_ka_dB  = fspl_dB(d_mars_earth, f_ka);
L_fs_uhf_dB = fspl_dB(d_surface_orbiter, f_uhf);

%% Received Power Calculation Function
rx_power_dBm = @(Ptx,Gtx,Lfs,Lmisc,Grx) ...
    (10*log10(Ptx*1e3) + Gtx - Lfs - Lmisc + Grx);

% Calculate Received Powers
P_rx_x_dBm   = rx_power_dBm(P_tx_x_W, G_tx_x, L_fs_x_dB, L_misc_ds, G_rx_x);
P_rx_ka_dBm  = rx_power_dBm(P_tx_ka_W, G_tx_ka, L_fs_ka_dB, L_misc_ds, G_rx_ka);
P_rx_uhf_dBm = rx_power_dBm(P_tx_surface_W,G_tx_surface,L_fs_uhf_dB,L_misc_surf,G_rx_orbiter);

%% Receiver Sensitivities
receiver_sensitivity_x   = -130; % X-band DSN receiver sensitivity [dBm]
receiver_sensitivity_ka  = -130; % Ka-band DSN receiver sensitivity [dBm]
receiver_sensitivity_uhf = -110; % UHF Electra payload receiver sensitivity [dBm]

%% Link Margin Calculation
link_margin_x   = P_rx_x_dBm - receiver_sensitivity_x;
link_margin_ka  = P_rx_ka_dBm - receiver_sensitivity_ka;
link_margin_uhf = P_rx_uhf_dBm - receiver_sensitivity_uhf;

%% Display Results
fprintf('\n--- Link Budget Summary with Link Margins ---\n');
fprintf('%-12s %-20s %-12s %-12s %-12s\n', 'Link', 'Received Power(dBm)', 'Sensitivity(dBm)', 'Link Margin(dB)', 'Status');
fprintf('%-12s %-20.2f %-12.2f %-12.2f %-12s\n', ...
    'X-band', P_rx_x_dBm, receiver_sensitivity_x, link_margin_x, ...
    string(link_margin_x > 0));
fprintf('%-12s %-20.2f %-12.2f %-12.2f %-12s\n', ...
    'Ka-band', P_rx_ka_dBm, receiver_sensitivity_ka, link_margin_ka, ...
    string(link_margin_ka > 0));
fprintf('%-12s %-20.2f %-12.2f %-12.2f %-12s\n', ...
    'UHF Relay', P_rx_uhf_dBm, receiver_sensitivity_uhf, link_margin_uhf, ...
    string(link_margin_uhf > 0));