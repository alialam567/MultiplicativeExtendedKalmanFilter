clc;
clear;


[ax, ay, az, gx, gy, gz, time_imu] = load_imu_data('DroneIMUData2.mat');
[px, py, pz, roll, pitch, yaw, vx, vy, vz, time_pose] = load_pose_data('DronePoseData2.mat');

dt = 0.000977;
% t = 0:dt:40;
% 
% gt_dt = dt*32;
% gt_t = 0:gt_dt:40;

t = time_imu;
gt_t = time_pose;

% Convert ground truth angles (ZYX) to match estimation format
eul_true_rad = deg2rad([yaw, pitch, roll]); % [deg] to [rad]
quat_true = eul2quat(eul_true_rad, 'ZYX');
eul_true = rad2deg(quat2eul(quat_true, 'ZYX')); % [rad] to [deg]

gyroData = [gx gy gz];
accelData = [ax ay az];

% Low-pass filter parameters
alpha = 0.001; % smoothing factor (0 < alpha < 1); lower = smoother

% Initialize filtered data
accelFiltered = zeros(size(accelData));
accelFiltered(1,:) = accelData(1,:); % initialize with first value

% Apply low-pass filter
for i = 2:length(accelData)
    accelFiltered(i,:) = alpha * accelData(i,:) + (1 - alpha) * accelFiltered(i-1,:);
end

% Initial state
% Your initial Euler angles in degrees
initialEulerDeg = [0.5, 0, 0];

% Convert to radians
initialEulerRad = deg2rad(initialEulerDeg);

% Convert to quaternion (ZYX convention: yaw-pitch-roll)
initialQuat = eul2quat(initialEulerRad, 'ZYX');
%initialQuat = [1 0 0 0]; % [w x y z]
initialCov = 1e-5;

gyro_noise_density = 0.0028; % [dps/sqrt(Hz)], example value
gyro_noise_std = gyro_noise_density * sqrt(1/dt); % [dps]
gyro_cov = (gyro_noise_std * pi/180)^2; % convert to [rad^2] and square

accel_noise_density = 0.07; % [mg/sqrt(Hz)], example value
g = 9.71; % gravity [m/s^2]
accel_noise_std = accel_noise_density * 1e-3 * g * sqrt(1/dt); % [m/s^2]
accel_cov = accel_noise_std^2; % [m^2/s^4]

kf = MEKF_Ali(initialQuat, initialCov, ...
    gyro_cov, 1e-3, accel_cov, 1e-8, 0.01);

% Preallocate outputs
q_gyro_only = initialQuat;
quat_est = zeros(length(t), 4);
quat_gyro = zeros(length(t), 4);
quat_true = zeros(length(t), 4);

q_current = quaternion(initialQuat);

for i = 1:length(t)
    % Gyro-only integration
    omega_t = gyroData(i,:)'; % 3x1
    omega_quat = [0; omega_t]; % pure quaternion
    q_dot = 0.5 * quatmultiply(q_gyro_only, omega_quat');
    q_gyro_only = q_gyro_only + dt * q_dot;
    q_gyro_only = q_gyro_only / norm(q_gyro_only);

    % MEKF update
    acc_meas_corrected = accelFiltered;
    kf.update(gyroData(i,:)', acc_meas_corrected(i,:)', dt);

    % Store results
    quat_est(i,:) = kf.estimate;
    quat_gyro(i,:) = q_gyro_only;
end

% Convert all to Euler angles
eul_gyro = quat2eul(quat_gyro, 'ZYX');
eul_est = quat2eul(quat_est, 'ZYX');

% Convert from radians to degrees
eul_gyro = rad2deg(eul_gyro);
eul_est = rad2deg(eul_est);

rmse_mekf = calculateRMSE(gt_t, eul_true, t, eul_est);
rmse_gyro = calculateRMSE(gt_t, eul_true, t, eul_gyro);

fprintf('MEKF RMSE (deg): [%.2f, %.2f, %.2f]\n', rmse_mekf);
fprintf('Gyro RMSE (deg): [%.2f, %.2f, %.2f]\n', rmse_gyro);

% Plot orientation estimation vs ground truth
figure;
labels = {'Yaw', 'Pitch', 'Roll'};
for i = 1:3
    subplot(3,1,i);
    plot(t, eul_gyro(:,i), 'r', 'LineWidth', 1.5); hold on;
    plot(t, eul_est(:,i), 'b', 'LineWidth', 1.5);
    plot(gt_t, eul_true(1:length(gt_t),i), 'k--', 'LineWidth', 1.5); % Ground truth
    
    ylabel([labels{i} ' (deg)'], 'FontSize', 16);
    ax = gca;
    ax.FontSize = 14;  % Axis ticks and labels
    grid on;

    if i == 1
        legend('Gyro Only', 'MEKF', 'Ground Truth', 'FontSize', 14, 'Location', 'best');
        title('Orientation Estimation vs Ground Truth', 'FontSize', 18);
    end
end
xlabel('Time (s)', 'FontSize', 16);
% Plot estimated velocity and ground truth
if isprop(kf, 'velocity_log') && ~isempty(kf.velocity_log)
    figure;
    subplot(3,1,1);
    plot(t, kf.velocity_log(1,:), 'b', 'LineWidth', 1.2); hold on;
    plot(gt_t, vx, 'k--', 'LineWidth', 1.2);
    title('Estimated Velocity (X)'); ylabel('m/s'); grid on;

    subplot(3,1,2);
    plot(t, kf.velocity_log(2,:), 'b', 'LineWidth', 1.2); hold on;
    plot(gt_t, vy, 'k--', 'LineWidth', 1.2);
    title('Estimated Velocity (Y)'); ylabel('m/s'); grid on;

    subplot(3,1,3);
    plot(t, kf.velocity_log(3,:), 'b', 'LineWidth', 1.2); hold on;
    plot(gt_t, vz, 'k--', 'LineWidth', 1.2);
    title('Estimated Velocity (Z)'); ylabel('m/s'); grid on;
    xlabel('Time (s)');
    legend('MEKF Estimate', 'Ground Truth');
else
    warning('velocity_log not found in MEKF object. Skipping velocity plot.');
end

% Plot estimated velocity and ground truth
if isprop(kf, 'position_log') && ~isempty(kf.position_log)
    figure;
    % subplot(3,1,1);
    % plot(t, kf.position_log(1,:), 'b', 'LineWidth', 1.2); hold on;
    % plot(gt_t, px, 'k--', 'LineWidth', 1.2);
    % title('Estimated Position (X)'); ylabel('m'); grid on;
    % 
    % subplot(3,1,2);
    % plot(t, kf.position_log(2,:), 'b', 'LineWidth', 1.2); hold on;
    % plot(gt_t, py, 'k--', 'LineWidth', 1.2);
    % title('Estimated Position (Y)'); ylabel('m'); grid on;
    plot(t, kf.position_log(3,:), 'b', 'LineWidth', 1.2); hold on;
    plot(gt_t, pz, 'k--', 'LineWidth', 1.2);
    title('Estimated Position (Z)'); ylabel('m'); grid on;
    xlabel('Time (s)');
    legend('MEKF Estimate', 'Ground Truth');
else
    warning('velocity_log not found in MEKF object. Skipping velocity plot.');
end

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

function rmse = calculateRMSE(gt_t, eul_true, t, eul_est)
    % Find the exact matching timestamps between gt_t and t
    [common_t, idx_gt, idx_est] = intersect(gt_t, t);
    
    % Extract the matching Euler angles for both true and estimated data
    matched_eul_true = eul_true(idx_gt, :);
    matched_eul_est = eul_est(idx_est, :);

    % Calculate RMSE for each axis (Yaw, Pitch, Roll)
    diff = matched_eul_true - matched_eul_est;
    squared_errors = diff.^2;
    squared_errors
    mse = mean(squared_errors, 1);
    mse
    rmse = sqrt(mse);
end