clc;
clear;

% Time vector
dt = 0.01;
t = 0:dt:100;

% Simulated angular velocity (rad/s) — constant rotation around X-axis
angularVel = repmat([0.001, 0.001, 0.001], length(t), 1); % [rad/s]

% Noise and bias settings
gyroNoiseStd = 0.001;
accelNoiseStd = 0.01;
gyroDriftStd = 5e-3;
accelDriftStd = 1e-9;

% Initial orientation (quaternion)
initialQuat = [1 0 0 0]; % [w x y z]
initialCov = 1e-6;

% Create MEKF
kf = MEKF_Ali(initialQuat, initialCov, ...
    1e-4, 1e-4, 1e-3, 1e-9, 0.00000001);

% Preallocate arrays
quat_true = zeros(length(t), 4);
quat_gyro = zeros(length(t), 4);
quat_est = zeros(length(t), 4);
q_gyro_only = initialQuat;
q_current = quaternion(initialQuat);

% Gravity vector in world frame
gravity_world = [0, 0, -9.81];
accelMeas = zeros(length(t), 3);

% Simulate true quaternion and accelerometer data
for i = 1:length(t)
    % Integrate true quaternion using angular velocity
    omega_true = angularVel(i,:)';
    omega_quat_true = [0; omega_true];
    q_dot_true = 0.5 * quatmultiply(compact(q_current), omega_quat_true');
    q_current = quaternion(compact(q_current) + dt * q_dot_true);
    q_current = normalize(q_current);
    quat_true(i,:) = compact(q_current);
    
    % Rotate gravity from world to body frame to simulate accelerometer
    accelMeas(i,:) = rotateframe(conj(q_current), gravity_world);
end

% Simulate sensor readings (with bias and noise)
gyroBias = randn(1, 3) * gyroDriftStd;
gyroData = angularVel + gyroBias + gyroNoiseStd * randn(size(angularVel));

accelBias = randn(1, 3) * accelDriftStd;
accelData = accelMeas + accelBias + accelNoiseStd * randn(size(accelMeas));

% Run MEKF, gyro-only estimation, and Madgwick filter
for i = 1:length(t)
    % Gyro-only integration
    omega_t = gyroData(i,:)';
    omega_quat = [0; omega_t];
    q_dot = 0.5 * quatmultiply(q_gyro_only, omega_quat');
    q_gyro_only = q_gyro_only + dt * q_dot;
    q_gyro_only = q_gyro_only / norm(q_gyro_only);
    quat_gyro(i,:) = q_gyro_only;

    % MEKF update
    kf.update(gyroData(i,:)', accelData(i,:)', dt);
    quat_est(i,:) = kf.estimate;
end

% Convert quaternions to Euler angles (ZYX convention)
eul_true = quat2eul(quat_true, 'ZYX');
eul_gyro = quat2eul(quat_gyro, 'ZYX');
eul_est = quat2eul(quat_est, 'ZYX');

% Convert from radians to degrees
eul_true = rad2deg(eul_true);
eul_gyro = rad2deg(eul_gyro);
eul_est = rad2deg(eul_est);

% Plot results
figure;
labels = {'Yaw', 'Pitch', 'Roll'};
for i = 1:3
    subplot(3,1,i);
    plot(t, eul_true(:,i), 'k--', 'LineWidth', 1.2); hold on;
    plot(t, eul_gyro(:,i), 'r', 'LineWidth', 1.2);
    plot(t, eul_est(:,i), 'b', 'LineWidth', 1.2);
    
    ylabel([labels{i} ' (deg)'], 'FontSize', 14);
    grid on;
    
    ax = gca;
    ax.FontSize = 12;  % Set font size for axis ticks and labels
    
    if i == 1
        legend('True', 'Gyro Only', 'MEKF', 'FontSize', 12);
        title('Orientation Estimation', 'FontSize', 16);
    end
end
xlabel('Time (s)', 'FontSize', 14);
% Assuming your MEKF object is called `filter`
time = (0:length(kf.h_predicted_log)-1) * dt;
figure;

subplot(3,1,1);
plot(time, kf.h_predicted_log');
title('h\_predicted', 'FontSize', 14);
legend('x', 'y', 'z', 'FontSize', 12);
ylabel('Value', 'FontSize', 12);

subplot(3,1,2);
plot(time, kf.acc_meas_norm_log');
title('acc\_meas\_norm', 'FontSize', 14);
legend('x', 'y', 'z', 'FontSize', 12);
ylabel('Value', 'FontSize', 12);

subplot(3,1,3);
plot(time, kf.delta_theta_log');
title('\delta\theta', 'FontSize', 14);
legend('\delta\theta_x', '\delta\theta_y', '\delta\theta_z', 'FontSize', 12);
xlabel('Time (s)', 'FontSize', 12);
ylabel('Radians', 'FontSize', 12);


% Plot estimated velocity (if available)
if isprop(kf, 'velocity_log') && ~isempty(kf.velocity_log)
    figure;
    subplot(3,1,1);
    plot(time, kf.velocity_log(1,:));
    title('Estimated Velocity (X)'); ylabel('m/s'); grid on;
    
    subplot(3,1,2);
    plot(time, kf.velocity_log(2,:));
    title('Estimated Velocity (Y)'); ylabel('m/s'); grid on;
    
    subplot(3,1,3);
    plot(time, kf.velocity_log(3,:));
    title('Estimated Velocity (Z)'); ylabel('m/s'); grid on;
    xlabel('Time (s)');
else
    warning('velocity_log not found in MEKF object. Skipping velocity plot.');
end

% % Plot accelerometer measurement
% figure;
% subplot(3,1,1);
% plot(t, accelData(:,1));
% title('Measured Acceleration (X)'); ylabel('m/s^2'); grid on;
% 
% subplot(3,1,2);
% plot(t, accelData(:,2));
% title('Measured Acceleration (Y)'); ylabel('m/s^2'); grid on;
% 
% subplot(3,1,3);
% plot(t, accelData(:,3));
% title('Measured Acceleration (Z)'); ylabel('m/s^2'); grid on;
% xlabel('Time (s)');

