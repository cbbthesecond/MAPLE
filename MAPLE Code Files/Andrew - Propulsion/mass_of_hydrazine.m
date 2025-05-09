clc;clear all;
g_earth = 9.81;

MR_107_FORCE = 296 * 6; %296 N of thrust, 6 thrusters
MR_106_FORCE = 34 * 6; %34 N of thrust, 7 thrusters

MR_107_ISP = 230;
MR_106_ISP = 235;
MR_103_ISP = 215;

mass_flow_MR_107 =  MR_107_FORCE / (MR_107_ISP * g_earth);
mass_flow_MR_106 =  MR_106_FORCE / (MR_106_ISP * g_earth);

delta_V_stationkeeping = 50;
delta_V_circularization = 45.09;
delta_V_aerobrake = 2.99;
delta_V_initial_positioning = 861.1;

orbiter_dry_mass = 590.84; %kg
probe_mass = 448.97/2; %kg
density_hydrazine = 1032; %kg/m^3

%after all orbital maneuvers have been completed, use dry mass
pre_stationkeeping_mass = orbiter_dry_mass * (exp(delta_V_stationkeeping / (MR_103_ISP*g_earth)));
mass_hydrazine_stationkeeping = pre_stationkeeping_mass - orbiter_dry_mass;
volume_hydrazine_stationkeeping = mass_hydrazine_stationkeeping / density_hydrazine;

fprintf('Mass before all stationkeeping burns = %f kg\n',pre_stationkeeping_mass)
fprintf('Mass after all stationkeeping burns = %f kg\n',orbiter_dry_mass)
fprintf('Mass of hydrazine burned for stationkeeping = %f kg\n', mass_hydrazine_stationkeeping)
fprintf('Volume of hydrazine burned for stationkeeping = %f m^3\n\n', volume_hydrazine_stationkeeping)

%before any station keeping, need to get into final circular orbit
pre_circularization_mass = ...
    (pre_stationkeeping_mass+(2*probe_mass)) * (exp(delta_V_circularization / (MR_106_ISP*g_earth)));
mass_hydrazine_circularization = ...
    pre_circularization_mass - (pre_stationkeeping_mass + (2*probe_mass));
volume_hydrazine_circularization = mass_hydrazine_circularization / density_hydrazine;

time_of_burn_MR_106_circ = mass_hydrazine_circularization / mass_flow_MR_106;

fprintf('Mass before circularization burns = %f kg\n',pre_circularization_mass)
fprintf('Mass after circularization burns = %f kg\n',pre_stationkeeping_mass + (2*probe_mass))
fprintf('Mass of hydrazine burned for circularization = %f kg\n', mass_hydrazine_circularization)
fprintf('Volume of hydrazine burned for circularization = %f m^3\n', volume_hydrazine_circularization)
fprintf('Burn time of MR-106L thruster = %f s\n\n', time_of_burn_MR_106_circ)

%small aerobraking maneuvers
pre_aerobrake_mass = ...
    pre_circularization_mass * (exp(delta_V_aerobrake / (MR_106_ISP*g_earth)));
mass_hydrazine_aerobrake = ...
    pre_aerobrake_mass - pre_circularization_mass;
volume_hydrazine_aerobrake = mass_hydrazine_aerobrake / density_hydrazine;

time_of_burn_MR_106_aerobrake = mass_hydrazine_aerobrake / mass_flow_MR_106;

fprintf('Mass before aerobrake burns = %f kg\n',pre_aerobrake_mass)
fprintf('Mass after aerobrake burns = %f kg\n',pre_circularization_mass)
fprintf('Mass of hydrazine burned for aerobraking = %f kg\n', mass_hydrazine_aerobrake)
fprintf('Volume of hydrazine burned for aerobrake = %f m^3\n', volume_hydrazine_aerobrake)
fprintf('Burn time of MR-106L thruster = %f s\n\n', time_of_burn_MR_106_aerobrake)

%_________main alignment for aerobrake/capture manauver______________________
pre_positioning_mass = ...
    pre_aerobrake_mass * (exp(delta_V_initial_positioning / (MR_107_ISP*g_earth)));
mass_hydrazine_positioning = ...
    pre_positioning_mass - pre_aerobrake_mass;
volume_hydrazine_positioning = mass_hydrazine_positioning / density_hydrazine;

mass_for_MR107 = 0.3795 * mass_hydrazine_positioning;
mass_for_MR106 = 0.6205 * mass_hydrazine_positioning;

time_of_burn_MR_107_pos = mass_for_MR107 / mass_flow_MR_107;
time_of_burn_MR_106_pos = mass_for_MR106 / mass_flow_MR_106;

fprintf('Mass before positioning burns = %f kg\n',pre_positioning_mass)
fprintf('Mass after positioning burns = %f kg\n',pre_aerobrake_mass)
fprintf('Mass of hydrazine burned for initial positioning = %f kg\n', mass_hydrazine_positioning)
fprintf('Volume of hydrazine burned for positioning = %f m^3\n', volume_hydrazine_positioning)
fprintf('Time of MR107 burn = %f seconds or %f minutes\n', time_of_burn_MR_107_pos, time_of_burn_MR_107_pos/60)
fprintf('Time of MR106 burn = %f seconds or %f minutes\n\n', time_of_burn_MR_106_pos, time_of_burn_MR_106_pos/60)

%totals

total_burn_time_MR106 = time_of_burn_MR_106_aerobrake + time_of_burn_MR_106_circ + time_of_burn_MR_106_pos;
total_burn_time_MR107 = time_of_burn_MR_107_pos;
total_hydrazine_mass = ....
    mass_hydrazine_positioning + mass_hydrazine_aerobrake + ...
    mass_hydrazine_circularization + mass_hydrazine_stationkeeping;
mass_hydrazine_needed = ...
    total_hydrazine_mass + 0.15*total_hydrazine_mass;
volume_hydrazine_needed = total_hydrazine_mass / density_hydrazine;
fprintf('Total MR106L burn time: %f\n', total_burn_time_MR106)
fprintf('Total MR107N burn time: %f\n', total_burn_time_MR107)
fprintf('Total hydrazine mass expended = %f kg\n',total_hydrazine_mass)
fprintf('Total mass hydrazine required (expended + 15%% contingency) = %f kg\n', mass_hydrazine_needed)
fprintf('Total volume hydrazine required = %f m^3\n', volume_hydrazine_needed)
fprintf('Total volume hydrazine required (with 15%% contingency) = %f m^3\n', volume_hydrazine_needed + 0.15*volume_hydrazine_needed)
fprintf('Total Fueled Weight of Orbiter + Probes = %f kg', mass_hydrazine_needed + orbiter_dry_mass + probe_mass*2)