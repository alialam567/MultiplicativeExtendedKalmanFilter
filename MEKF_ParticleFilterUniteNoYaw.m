%% MEKF_ParticleFilterUnite.m
% author: Samuel Street
% date: 2025-04-12

% performs testing of localization using only a positional ekf and then
% using this result along with the particle filter to improve the results

close all
clear
clc

%% data loading
% loads the data produced by the simulation
load("drone_sensor_config.mat")
%load("drone_sim_log.mat")
load("DATAREALFR.mat")
load("likelihood_field.mat")

data_log_arr = table2array(data_log_table);

tof_data = [data_log_table.tof1'; data_log_table.tof2' ; data_log_table.tof3' ; data_log_table.tof4'];

%% Setup MEKF

[ax, ay, az, gx, gy, gz, time_imu] = load_imu_data('DroneIMUData2.mat');
[px, py, pz, roll, pitch, yaw, vx, vy, vz, time_pose] = load_pose_data('DronePoseData2.mat');

dt = 0.000977;

t = time_imu;
gt_t = time_pose;

% Convert ground truth angles (ZYX) to match estimation format
eul_true_rad = deg2rad([yaw, pitch, roll]); % [deg] to [rad]
quat_true = eul2quat(eul_true_rad, 'ZYX');
eul_true = rad2deg(quat2eul(quat_true, 'ZYX')); % [rad] to [deg]

gyroData = [gx gy gz];
accelData = [ax ay az];

% Low-pass filter parameters
alpha = 0.01; % smoothing factor (0 < alpha < 1); lower = smoother

% Initialize filtered data
accelFiltered = zeros(size(accelData));
accelFiltered(1,:) = accelData(1,:); % initialize with first value

% Apply low-pass filter
for i = 2:length(accelData)
    accelFiltered(i,:) = alpha * accelData(i,:) + (1 - alpha) * accelFiltered(i-1,:);
end

% Initial state
initialEulerDeg = [0, 0, 0];
initialEulerRad = deg2rad(initialEulerDeg);
initialQuat = eul2quat(initialEulerRad, 'ZYX');
initialPos = [0 0 1];
initialVel = [0 0 0];


initialCov = 1e-5;

gyro_noise_density = 0.0028; % [dps/sqrt(Hz)], example value
gyro_noise_std = gyro_noise_density * sqrt(1/dt); % [dps]
gyro_cov = (gyro_noise_std * pi/180)^2; % convert to [rad^2] and square

accel_noise_density = 0.07; % [mg/sqrt(Hz)], example value
g = 9.71; % gravity [m/s^2]
accel_noise_std = accel_noise_density * 1e-3 * g * sqrt(1/dt); % [m/s^2]
accel_cov = accel_noise_std^2; % [m^2/s^4]

% Create MEKF
% kf = MEKF_Ali(initialQuat, initialCov, ...
%     1e-5, 1e-3, 1e-2, 1e-8, 0.05);

kf = MEKF_TOF(initialQuat, initialPos, initialVel, initialCov, ...
    gyro_cov, 1e-3, accel_cov, 1e-8, 0.01, 1e-5, 1e-8, 0.001);

% Preallocate outputs
q_gyro_only = initialQuat;
quat_est = zeros(length(t), 4);
quat_gyro = zeros(length(t), 4);
quat_true = zeros(length(t), 4);

q_current = quaternion(initialQuat);

%% PF Particle Setup

% for updating the particles
f_pf = @(particles, x) particles + (x(1:3)-mean(particles,2)) + x(4:6)*dt + 0.5*x(7:9)*dt^2;
pf_update_noise = 0.25*eye(3);

% Initialize PF parameters
pf_likelihood_closenes_tol = 0.1;
pf_regularizer = 0;
pf_equiv_tol = 1e-3;
pf_weight_tol = 1e-3;

num_particles = 500;
R_pf = 0.30^2*eye(3);

pf_particles = mvnrnd(initialPos', R_pf, num_particles)';

%% run this ekf with accel and pf

for i=1:10000 % CHANGE THIS FOR FULL DATA

    % update the state of the estimation
    acc_meas_corrected = accelFiltered;
    kf.predict(gyroData(i,:)', acc_meas_corrected(i,:)', dt);
    
    orientation = quat2eul(kf.estimate, 'ZYX');
    % update the particle filter

    xk_use_in_pf = [kf.position', kf.velocity', acc_meas_corrected(i,:)];

    pf_particles = f_pf(pf_particles, xk_use_in_pf');

    pf_weights = compute_pos_particle_weights(pf_particles, pf_regularizer, ...
        pf_equiv_tol, drone_tof_sensors, tof_data(:,i), ...
        pf_likelihood_closenes_tol, likelihood_field, likelihood_xlocs, ...
        likelihood_ylocs, likelihood_zlocs);
    pf_particles = redistribute_pos_particles_under_weight(pf_particles, ...
        pf_weights, pf_weight_tol, pf_update_noise);
    pf_pos_est = mean(pf_particles,2);

    kf.update(acc_meas_corrected(i,:)',pf_pos_est')

    quat_est(i,:) = kf.estimate;
    
    pf_pos_est
end

%% Plotting

% Convert all to Euler angles
eul_est = quat2eul(quat_est, 'ZYX');

% Convert from radians to degrees
eul_est = rad2deg(eul_est);

% Plot  
figure;
labels = {'Yaw', 'Pitch', 'Roll'};
for i = 1:3
    subplot(3,1,i);
    plot(t, eul_est(:,i), 'b', 'LineWidth', 1.2);
    ylabel([labels{i} ' (deg)']);
    grid on;
    if i == 1
        legend('MEKF');
        title('Orientation Estimation vs Ground Truth');
    end
end
xlabel('Time (s)');

% % Plot estimated velocity and ground truth
% if isprop(kf, 'velocity_log') && ~isempty(kf.velocity_log)
%     figure;
%     subplot(3,1,1);
%     plot(1:10, kf.velocity_log(1,:), 'b', 'LineWidth', 1.2); hold on;
%     plot(1:2007, vx, 'k--', 'LineWidth', 1.2);
%     title('Estimated Velocity (X)'); ylabel('m/s'); grid on;
% 
%     subplot(3,1,2);
%     plot(1:500, kf.velocity_log(2,:), 'b', 'LineWidth', 1.2); hold on;
%     plot(1:2007, vy, 'k--', 'LineWidth', 1.2);
%     title('Estimated Velocity (Y)'); ylabel('m/s'); grid on;
% 
%     subplot(3,1,3);
%     plot(1:500, kf.velocity_log(3,:), 'b', 'LineWidth', 1.2); hold on;
%     plot(1:2007, vz, 'k--', 'LineWidth', 1.2);
%     title('Estimated Velocity (Z)'); ylabel('m/s'); grid on;
%     xlabel('Time (s)');
%     legend('MEKF Estimate', 'Ground Truth');
% else
%     warning('velocity_log not found in MEKF object. Skipping velocity plot.');
% end

% Plot estimated velocity and ground truth
if isprop(kf, 'position_log') && ~isempty(kf.position_log)
    figure;
    % subplot(3,1,1);
    % plot(1:10000, kf.position_log(1,:), 'b', 'LineWidth', 1.2); hold on;
    % plot(1:2007, px, 'k--', 'LineWidth', 1.2);
    % title('Estimated Position (X)'); ylabel('m'); grid on;
    % 
    % subplot(3,1,2);
    % plot(1:10000, kf.position_log(2,:), 'b', 'LineWidth', 1.2); hold on;
    % plot(1:2007, py, 'k--', 'LineWidth', 1.2);
    % title('Estimated Position (Y)'); ylabel('m'); grid on;

    subplot(3,1,3);
    plot(1:10000, kf.position_log(3,:), 'b', 'LineWidth', 1.2); hold on;
    %plot(1:330, pz, 'k--', 'LineWidth', 1.2);
    title('Estimated Position (Z)'); ylabel('m'); grid on;
    xlabel('Time (s)');
    legend('MEKF Estimate', 'Ground Truth');
else
    warning('velocity_log not found in MEKF object. Skipping velocity plot.');
end


%% Data Loading Functions

function [ax,ay,az,gx,gy,gz,timestamp] = load_imu_data(filename)
    load(filename, "imu1")  % Load table
    
    % Extract data using table column names
    ax = imu1.accl_x;
    ay = imu1.accl_y;
    az = imu1.accl_z;
    
    gx = imu1.gyro_x;
    gy = imu1.gyro_y;
    gz = imu1.gyro_z;

    timestamp = imu1.timestamp_ms;
end

function [px, py, pz, roll, pitch, yaw, vx, vy, vz, timestamp] = load_pose_data(filename)
    load(filename, "pose1")  % Load table
    
    % Extract data using table column names
    px = pose1.pos_x;
    py = pose1.pos_y;
    pz = pose1.pos_z;

    roll = pose1.roll_deg - 0.5;
    pitch = pose1.pitch_deg + 1.7;
    yaw = pose1.yaw_deg;
    
    vx = pose1.vel_x;
    vy = pose1.vel_y;
    vz = pose1.vel_z;

    timestamp = pose1.timestamp_ms;
end