close; clear; clc;
%% Intro
%This is the sizing code for the orbiter. Sizing is done for the most
%intensive phase of the mission-during science orbit.
%
%Parameters describe the mission and define the characteristics of the
%solar panels and battery.
%
%Each component needs a two vector pair: Power_Required_W and Time_On_MIN:
%element i of Power_Required_W is the power the component uses for the
%length of time defined by element i of Time_On_MIN

%% Parameters

%Mission
Orbital_Period_MIN = 117.41; %[min]
Eclipse_Period_MIN = 40; %[min]
Mission_Lifetime_YRS = 10; %[years]
Solar_Irradiance_WMs = 1367; %[W/m^2] - at 1 AU
Irradiance_Factor = 0.431; %reduction in irradiance due to distance at Mars
Design_Margin_PER = 10; %[%]

%Solar Array
Panel_Efficiency_PER = 28; %[%]
Degradation_Rate_PPerYR = 3.75; %[% per year]
Pointing_Efficiency_PER = 100; %[%]
Solar_Array_Mass_Per_Area_KGMs = 4; %[kg/m^2]

%Battery
Cycle_Life_CYCLES = 16000; %[cycle life at 100% DOD]
Conversion_Efficiency_PER = 85; %[%]
Specific_Energy_WhKG = 160; %[Wh/kg]
Energy_Density_WhL = 393; %[Wh/L]

%% Define Power Profile

%Instruments
Orbiter.Instruments.MARLI.Power_Required_W = 81;
Orbiter.Instruments.MARLI.Time_On_MIN = Orbital_Period_MIN;

Orbiter.Instruments.SPICAM.Power_Required_W = 26;
Orbiter.Instruments.SPICAM.Time_On_MIN = Orbital_Period_MIN;

Orbiter.Instruments.MOLA.Power_Required_W = 34.2;
Orbiter.Instruments.MOLA.Time_On_MIN = Orbital_Period_MIN;

%Propulsion
%-station keeping, desaturation
Orbiter.Propulsion.Station_Keeping_Valves.Power_Required_W = [0 80 0];
Orbiter.Propulsion.Station_Keeping_Valves.Time_On_MIN = [30 2 Orbital_Period_MIN-30-2];

Orbiter.Propulsion.Desaturation_Valves.Power_Required_W = [0 80 0];
Orbiter.Propulsion.Desaturation_Valves.Time_On_MIN = [Orbital_Period_MIN./2 1 Orbital_Period_MIN./2-1];

%ADCS
%-reaction wheels
Orbiter.ADCS.Reaction_Wheels.Power_Required_W = [35 100 65 100];
Orbiter.ADCS.Reaction_Wheels.Time_On_MIN = [10 Orbital_Period_MIN./2-10 10 Orbital_Period_MIN./2-10];

%Comms
%-transmitting to earth, receiving from surface
Orbiter.Comms.XBand_Antenna.Power_Required_W = [62 716 62];
Orbiter.Comms.XBand_Antenna.Time_On_MIN = [10 20 Orbital_Period_MIN-30];

%Orbiter.Comms.XBand_Antenna.Power_Required_W = [716 62 62];
%Orbiter.Comms.XBand_Antenna.Time_On_MIN = [20 10 Orbital_Period_MIN-30];

Orbiter.Comms.UHF_Receiver.Power_Required_W = [6.6 33 6.6 33 6.6 33 6.6];
Orbiter.Comms.UHF_Receiver.Time_On_MIN = [40 10 15 10 15 10 Orbital_Period_MIN-100];

%Thermal
%-radiators,heaters
Orbiter.Thermal.Radiator.Power_Required_W = [16.5 40];
Orbiter.Thermal.Radiator.Time_On_MIN = [Orbital_Period_MIN-Eclipse_Period_MIN Eclipse_Period_MIN];

Orbiter.Thermal.Heater.Power_Required_W = [0 2];
Orbiter.Thermal.Heater.Time_On_MIN = [Orbital_Period_MIN-Eclipse_Period_MIN Eclipse_Period_MIN];

%Power
%-active power regulation system, sun tracking system
Orbiter.Power.Power_Management.Power_Required_W = 100;
Orbiter.Power.Power_Management.Time_On_MIN = Orbital_Period_MIN;

Orbiter.Power.Active_Solar_Array.Power_Required_W = 35;
Orbiter.Power.Active_Solar_Array.Time_On_MIN = Orbital_Period_MIN;

%Data Handling
%-data processing
Orbiter.CDH.Flight_Computer.Power_Required_W = 50;
Orbiter.CDH.Flight_Computer.Time_On_MIN = Orbital_Period_MIN;

%% Sizing
dt = 1; %[min]
t = 0:dt:Orbital_Period_MIN;

subsystems = fieldnames(Orbiter);
Orbiter.Power_Profile_W = zeros(size(t));
for i = 1:length(subsystems)
    subsystem = subsystems{i};
    components = fieldnames(Orbiter.(subsystem));

    Subsystem_Struct = Orbiter.(subsystem);
    Subsystem_Power_Profile = zeros(size(t));
    for j = 1:length(components)
        component = components{j};
        component_power_W = Subsystem_Struct.(component).Power_Required_W;
        component_time_on_MIN = Subsystem_Struct.(component).Time_On_MIN;
        component_power_profile = zeros(size(t));
        component_start_times = [0 cumsum(component_time_on_MIN)];
        for k = 2:length(component_start_times)
            if k ~= length(component_start_times)
                component_power_profile((t< component_start_times(k)) & (t>=component_start_times(k-1))) = component_power_profile(t< component_start_times(k) & t>=component_start_times(k-1)) + component_power_W(k-1);
            else
                component_power_profile((t<= component_start_times(k)) & (t>=component_start_times(k-1))) = component_power_profile(t<= component_start_times(k) & t>=component_start_times(k-1)) + component_power_W(k-1);
            end
            
        end
            Orbiter.(subsystem).(component).Power_Profile_W = component_power_profile;
            component_peak_power = max(component_power_profile);
            component_average_power = mean(component_power_profile);
            component_duty_cycle = component_average_power./component_peak_power;
            Orbiter.(subsystem).(component).Peak_Power = component_peak_power;
            Orbiter.(subsystem).(component).Average_Power = component_average_power;
            Orbiter.(subsystem).(component).Duty_Cycle = component_duty_cycle;
            Subsystem_Power_Profile = Subsystem_Power_Profile + component_power_profile;
    end

    Orbiter.(subsystem).Power_Profile_W = Subsystem_Power_Profile;
    subsystem_peak_power = max(Subsystem_Power_Profile);
    subsystem_average_power = mean(Subsystem_Power_Profile);
    subsystem_duty_cycle = subsystem_average_power./subsystem_peak_power;
    Orbiter.(subsystem).Peak_Power = subsystem_peak_power;
    Orbiter.(subsystem).Average_Power = subsystem_average_power;
    Orbiter.(subsystem).Duty_Cycle = subsystem_duty_cycle;
    Orbiter.Power_Profile_W = Orbiter.Power_Profile_W + Subsystem_Power_Profile;
end
Energy_Profile_Wh = Orbiter.Power_Profile_W .* dt./60;

%size batteries
Energy_Used_During_Eclipse_Wh = sum(Energy_Profile_Wh(t >= (Orbital_Period_MIN-Eclipse_Period_MIN)));
Usable_Battery_Capacity = Energy_Used_During_Eclipse_Wh./(Conversion_Efficiency_PER./100);
Required_Cycle_Life_CYCLES = Mission_Lifetime_YRS./(Orbital_Period_MIN./525600);
Depth_Of_Discharge_PER = Cycle_Life_CYCLES.*100./Required_Cycle_Life_CYCLES;

if Depth_Of_Discharge_PER > 80
    Depth_Of_Discharge_PER = 80;
end

Total_Battery_Capacity = Usable_Battery_Capacity./(Depth_Of_Discharge_PER./100);

Total_Battery_Capacity = Total_Battery_Capacity.*(1+Design_Margin_PER./100);

Orbiter.Battery.Capacity = Total_Battery_Capacity;
Orbiter.Battery.Depth_of_Discharge = Depth_Of_Discharge_PER;
Orbiter.Battery.Mass = Orbiter.Battery.Capacity./Specific_Energy_WhKG;
Orbiter.Battery.Volume = Orbiter.Battery.Capacity./Energy_Density_WhL;
Orbiter.Battery.Charging_Power = Usable_Battery_Capacity./((Orbital_Period_MIN-Eclipse_Period_MIN)./60);

%size solar panels
total_Energy = sum(Energy_Profile_Wh); %[Wh]
component_power_rqt = total_Energy./((Orbital_Period_MIN-Eclipse_Period_MIN)./60);

Solar_Array_Power_W = component_power_rqt;

Max_Power_During_Charging = max(Orbiter.Power_Profile_W(t <= (Orbital_Period_MIN-Eclipse_Period_MIN)));

if Solar_Array_Power_W < Max_Power_During_Charging
    Solar_Array_Power_W = Max_Power_During_Charging;
end

%the above equation only enforces the start and end conditions, we need to
%enforce that the battery actually charges to 100%
Min_Time_At_Full_Charge_MIN = 10;

Solar_Array_Energy_Prod_Charging_Wh = Solar_Array_Power_W.*(Orbital_Period_MIN-Eclipse_Period_MIN-Min_Time_At_Full_Charge_MIN)./60;
During_Charging_Energy_Consumed_Wh = sum(Energy_Profile_Wh(t<=(Orbital_Period_MIN-Eclipse_Period_MIN-Min_Time_At_Full_Charge_MIN)));
Energy_For_Charging_Wh = Solar_Array_Energy_Prod_Charging_Wh-During_Charging_Energy_Consumed_Wh;

if Energy_For_Charging_Wh < Usable_Battery_Capacity
    Additional_Energy_Needed_Wh = Usable_Battery_Capacity-Energy_For_Charging_Wh;
    Additional_Power_Needed_W = Additional_Energy_Needed_Wh./(Orbital_Period_MIN-Eclipse_Period_MIN-Min_Time_At_Full_Charge_MIN)*60;
    Solar_Array_Power_W = Solar_Array_Power_W + Additional_Power_Needed_W;
end

Solar_Array_Power_W = Solar_Array_Power_W.*(1+Design_Margin_PER./100);

n_degradation = (1-Degradation_Rate_PPerYR./100).^Mission_Lifetime_YRS;
n_pointing = Pointing_Efficiency_PER./100;
n_cell = Panel_Efficiency_PER./100;
R_Solar_Constant = Solar_Irradiance_WMs .* Irradiance_Factor;
Solar_Panel_Area_ms = Solar_Array_Power_W./n_degradation./n_pointing./n_cell./R_Solar_Constant; %[m^2]

Orbiter.Solar_Array.Output = Solar_Array_Power_W;
Orbiter.Solar_Array.Area = Solar_Panel_Area_ms;
Orbiter.Solar_Array.Mass = Solar_Array_Mass_Per_Area_KGMs.*Solar_Panel_Area_ms;

%test if solar array and batteries meet specifications
Solar_Panel_Output_W = zeros(size(t));
Solar_Panel_Output_W(t<(Orbital_Period_MIN-Eclipse_Period_MIN)) = Solar_Array_Power_W;
Battery_Capacity_Wh = zeros(size(t));
Battery_Capacity_Wh(1) = Total_Battery_Capacity-Usable_Battery_Capacity;
%Battery_Capacity_Wh(1) = Total_Battery_Capacity.*(1-Depth_Of_Discharge_PER./100);

Energy_Use_Wh = Energy_Profile_Wh - Solar_Panel_Output_W.*(dt./60);
for i = 2:length(t)
    Battery_Capacity_Wh(i) = Battery_Capacity_Wh(i-1) - Energy_Use_Wh(i);
    if Battery_Capacity_Wh(i) > Total_Battery_Capacity
        Battery_Capacity_Wh(i) = Total_Battery_Capacity;
    end
end
Battery_Percentage_Wh = Battery_Capacity_Wh./Total_Battery_Capacity;

Battery_Capacity_Steady_State_Wh = zeros(size(t));
Battery_Capacity_Steady_State_Wh(1) = Battery_Capacity_Wh(end);
for i = 2:length(t)
    Battery_Capacity_Steady_State_Wh(i) = Battery_Capacity_Steady_State_Wh(i-1) - Energy_Use_Wh(i);
    if Battery_Capacity_Steady_State_Wh(i) > Total_Battery_Capacity
        Battery_Capacity_Steady_State_Wh(i) = Total_Battery_Capacity;
    end
end
Battery_Percentage_Steady_State_Wh = Battery_Capacity_Steady_State_Wh./Total_Battery_Capacity;

sprintf('Power system sizing:\n Solar Array Power Output: %f W \n Solar Array Area: %f m^2 \n Solar Array mass: %f kg \n Battery Capacity: %f Wh \n Depth of Discharge: %f',Solar_Array_Power_W,Orbiter.Solar_Array.Area,Orbiter.Solar_Array.Mass,Orbiter.Battery.Capacity,Orbiter.Battery.Depth_of_Discharge)

figure(1);clf
plot(t,Orbiter.Power_Profile_W)
hold on
ylabel('Power Draw (W)')
xlabel('Time (min)')
xlim([min(t) max(t)])
grid on

figure(2);clf
plot(t,Battery_Percentage_Wh.*100)
%title('Battery Charge - From lowest start')
grid on
xlim([min(t) max(t)])
xlabel('Time (min)')
ylabel('Battery Charge (%)')

figure(3);clf
plot(t,Battery_Percentage_Steady_State_Wh*100)
%title('Battery Charge - Steady State Operations')
grid on
ylabel('Battery Charge (%)')
xlabel('Time (min)')
xlim([min(t) max(t)])
xlabel('Time (min)')
ylabel('Battery Charge (%)')

figure(4);clf
legend_vals = {};
for i = 1:length(subsystems)
    current_subsystem = subsystems{i};
    plot(t,Orbiter.(current_subsystem).Power_Profile_W)
    hold on
    legend_vals = [legend_vals, current_subsystem];
end
ylabel('Power Draw (W)')
xlabel('Time (min)')
legend(legend_vals)
grid on
xlim([min(t) max(t)])
