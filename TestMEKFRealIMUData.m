clc;
clear;

dt = 0.01;
t = 0:dt:29;

[ax, ay, az, gx, gy, gz] = load_imu_data('IMUDATA3.mat');

gyroData = [gx gy gz];
accelData = [ax ay az];

% Initial state
% Your initial Euler angles in degrees
initialEulerDeg = [0, 0, 0];

% Convert to radians
initialEulerRad = deg2rad(initialEulerDeg);

% Convert to quaternion (ZYX convention: yaw-pitch-roll)
initialQuat = eul2quat(initialEulerRad, 'ZYX');
%initialQuat = [1 0 0 0]; % [w x y z]
initialCov = 1e-6;

gyro_noise_density = 0.01; % [dps/sqrt(Hz)], example value
gyro_noise_std = gyro_noise_density * sqrt(1/dt); % [dps]
gyro_cov = (gyro_noise_std * pi/180)^2; % convert to [rad^2] and square

accel_noise_density = 0.16; % [mg/sqrt(Hz)], example value
g = 9.81; % gravity [m/s^2]
accel_noise_std = accel_noise_density * 1e-3 * g * sqrt(1/dt); % [m/s^2]
accel_cov = accel_noise_std^2; % [m^2/s^4]

% Create MEKF
% kf = MEKF_Ali(initialQuat, initialCov, ...
%     1e-5, 1e-3, 1e-2, 1e-8, 0.05);

kf = MEKF_Ali(initialQuat, initialCov, ...
    gyro_cov, 1e-3, accel_cov, 1e-8, 0.05);

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
    acc_meas_corrected = [accelData(:,1), accelData(:,2), accelData(:,3)];
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

% Plot  
figure;
labels = {'Yaw', 'Pitch', 'Roll'};
for i = 1:3
    subplot(3,1,i);
    plot(t, eul_gyro(:,i), 'r', 'LineWidth', 1.5);
    hold on;
    plot(t, eul_est(:,i), 'b', 'LineWidth', 1.5);
    
    ylabel([labels{i} ' (deg)'], 'FontSize', 16);
    grid on;

    ax = gca;
    ax.FontSize = 14;  % Bigger font for axis ticks and labels
    
    if i == 1
        legend('Gyro Only', 'MEKF', 'FontSize', 14, 'Location', 'best');
        title('Orientation Estimation vs Ground Truth', 'FontSize', 18);
    end
end
xlabel('Time (s)', 'FontSize', 16);

% Plot estimated velocity (if available)
if isprop(kf, 'velocity_log') && ~isempty(kf.velocity_log)
    figure;
    subplot(3,1,1);
    plot(t, kf.velocity_log(1,:));
    title('Estimated Velocity (X)'); ylabel('m/s'); grid on;
    
    subplot(3,1,2);
    plot(t, kf.velocity_log(2,:));
    title('Estimated Velocity (Y)'); ylabel('m/s'); grid on;
    
    subplot(3,1,3);
    plot(t, kf.velocity_log(3,:));
    title('Estimated Velocity (Z)'); ylabel('m/s'); grid on;
    xlabel('Time (s)');
else
    warning('velocity_log not found in MEKF object. Skipping velocity plot.');
end

function [ax,ay,az,gx,gy,gz] = load_imu_data(filename)
    load(filename, "seriallog20250410003647")  % Load table
    
    % Extract data using table column names
    ax = seriallog20250410003647.ax;
    ay = seriallog20250410003647.ay;
    az = seriallog20250410003647.az;
    
    gx = deg2rad(seriallog20250410003647.gx);
    gy = deg2rad(seriallog20250410003647.gy);
    gz = deg2rad(seriallog20250410003647.gz);
end