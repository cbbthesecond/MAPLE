% Monte Carlo Simulation for Budget Analysis Using Triangular and PERT Distributions
% and an Overlay Plot with Intersection Outputs
clear; clc; close all;

%% Input Parameters (Final Costs with Reserve)
final_mode = 986.4e6/1000000;       % Most Likely value (mode) with 10% reserve, in dollars
final_max  = 1e9/1000000;           % Maximum cost in dollars
reserve    = 0.10;          % 10% reserve built in

% Calculate the minimum (optimistic) value as the no-reserve cost.
final_min = final_mode / (1 + reserve);  % Approximately 896.73e6 dollars

% Display the three key points for reference
fprintf('Minimum (No Reserve): %g\n', final_min);
fprintf('Mode (With Reserve):  %g\n', final_mode);
fprintf('Maximum:              %g\n', final_max);

nIterations = 10000;  % Number of simulation iterations

%% --- Triangular Distribution Simulation ---
% The triangular distribution uses three parameters: a (min), b (mode), and c (max).
a = final_min;
b = final_mode;
c = final_max;

% Preallocate simulation vector for speed
triangularSamples = zeros(nIterations, 1);
p = (b - a) / (c - a);  % Partition point (probability that sample is below the mode)

% Generate samples from the triangular distribution using inverse transform sampling
U = rand(nIterations, 1);
for i = 1:nIterations
    if U(i) < p
        triangularSamples(i) = a + sqrt(U(i) * (b - a) * (c - a));
    else
        triangularSamples(i) = c - sqrt((1 - U(i)) * (c - b) * (c - a));
    end
end

% Sort samples and compute cumulative probabilities
sortedTri = sort(triangularSamples);
cumProbTri = (1:nIterations)' / nIterations;

%% Plot Triangular Distribution S-Curve
figure;
plot(sortedTri, cumProbTri * 100, 'b-', 'LineWidth', 2);  % Multiply by 100 for percentages
hold on;
% Mark the mode with a vertical dashed line.
plot([b, b], [0, 100], 'r--', 'LineWidth', 2);
xlabel('Cost ($)');
ylabel('Cumulative Probability (%)');
title('Budget Analysis S-Curve using Triangular Distribution');
legend('Triangular Simulation', 'Mode (Final Point Estimate)', 'Location', 'best');
grid on;
hold off;

%% --- PERT Distribution Simulation ---
% The PERT distribution is modeled as a Beta distribution scaled to [a, c].
% A common formulation defines the Beta shape parameters as:
%   alpha = ((b - a) * 4 / (c - a)) + 1
%   beta  = ((c - b) * 4 / (c - a)) + 1
alpha = ((b - a) * 4 / (c - a)) + 1;
beta_param  = ((c - b) * 4 / (c - a)) + 1;

% Generate samples from the Beta distribution and scale them to [a, c]
betaSamples = betarnd(alpha, beta_param, nIterations, 1);
pertSamples = a + betaSamples * (c - a);

% Sort samples and compute cumulative probabilities
sortedPERT = sort(pertSamples);
cumProbPERT = (1:nIterations)' / nIterations;

%% Plot PERT Distribution S-Curve
figure;
plot(sortedPERT, cumProbPERT * 100, 'm-', 'LineWidth', 2);  % Y-axis in percentages
hold on;
% Mark the mode as a vertical dashed line.
plot([b, b], [0, 100], 'r--', 'LineWidth', 2);
xlabel('Cost ($)');
ylabel('Cumulative Probability (%)');
title('Budget Analysis S-Curve using PERT Distribution');
legend('PERT Simulation', 'Mode (Final Point Estimate)', 'Location', 'best');
grid on;
hold off;

%% --- Overlay Plot of Triangular and PERT S-Curves ---
figure;
plot(sortedTri, cumProbTri * 100, 'b-', 'LineWidth', 2);    % Triangular S-curve
hold on;
plot(sortedPERT, cumProbPERT * 100, 'r-', 'LineWidth', 2);  % PERT S-curve
% Mark the mode with a vertical dashed line.
plot([b, b], [0, 100], 'k--', 'LineWidth', 2);

% Calculate the cumulative probability (in percentage) at the mode for each simulation:
cumProbAtMode_tri = sum(triangularSamples <= b) / nIterations * 100;
cumProbAtMode_pert = sum(pertSamples <= b) / nIterations * 100;

% Mark the intersections on the overlay plot:
plot(b, cumProbAtMode_tri, 'bo', 'MarkerSize',8, 'MarkerFaceColor','b');
plot(b, cumProbAtMode_pert, 'ro', 'MarkerSize',8, 'MarkerFaceColor','r');

% Annotate the plot with text labels showing the intersection percentages
text(b, cumProbAtMode_tri, sprintf('  %.1f%%', cumProbAtMode_tri), 'Color','b', 'FontSize',10, 'VerticalAlignment','bottom');
text(b, cumProbAtMode_pert, sprintf('  %.1f%%', cumProbAtMode_pert), 'Color','r', 'FontSize',10, 'VerticalAlignment','top');

xlabel('Cost ($M)');
ylabel('Cumulative Probability (%)');
legend('Triangular Simulation', 'PERT Simulation', 'Point Estimate', 'Location', 'best');
grid on;
hold off;

%% Output the Intersection Values
fprintf('Intersection at Mode (Triangular Simulation): %0.2f%%\n', cumProbAtMode_tri);
fprintf('Intersection at Mode (PERT Simulation):       %0.2f%%\n', cumProbAtMode_pert);
