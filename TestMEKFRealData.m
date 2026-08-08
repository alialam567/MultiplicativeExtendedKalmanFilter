clc;
clear;

dt = 0.005;
t = 0:dt:10;

data = readmatrix('data.csv');
save('data.mat', 'data');

data_gt = readmatrix('gt_data_r.csv');
save('data_gt.mat', 'data_gt');  % Fixed variable name consistency

[ax, ay, az, gx, gy, gz] = load_imu_data('data.mat');
[quat] = load_gt_data('data_gt.mat');
eul_gt = quat2eul(quat, 'ZYX');
eul_gt = rad2deg(eul_gt);
eul_gt = eul_gt(1:length(t),:);

gyroData = [gx gy gz];
accelData = [ax ay az];

% Initial state
% Your initial Euler angles in degrees
initialEulerDeg = [156.613165173185, -65.3628903115519, -176.214037161624];

% Convert to radians
initialEulerRad = deg2rad(initialEulerDeg);

% Convert to quaternion (ZYX convention: yaw-pitch-roll)
initialQuat = eul2quat(initialEulerRad, 'ZYX');
%initialQuat = [1 0 0 0]; % [w x y z]
initialCov = 1e-6;

% Create MEKF
kf = MEKF_Ali(initialQuat, initialCov, ...
    1e-4, 1e-4, 1e-2, 1e-5, 0.01);

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
    plot(t, eul_gyro(:,i), 'r', 'LineWidth', 1.2);
    hold on;
    plot(t, eul_est(:,i), 'b', 'LineWidth', 1.2);
    plot(t, eul_gt(:,i), 'k--', 'LineWidth', 1.2); % Ground truth as dashed black
    ylabel([labels{i} ' (deg)']);
    grid on;
    if i == 1
        legend('Gyro Only', 'MEKF', 'Ground Truth');
        title('Orientation Estimation vs Ground Truth');
    end
end
xlabel('Time (s)');

function [ax,ay,az,gx,gy,gz] = load_imu_data(filename)
    load(filename,"data")
    
    ax = data(:,5);
    ay = data(:,6);
    az = data(:,7);
    
    gx = data(:,2);
    gy = data(:,3);
    gz = data(:,4);
end

function [quat] = load_gt_data(filename)
    load(filename,"data_gt")  % Fixed variable name
    
    q1 = data_gt(:,5);  % Fixed variable name
    q2 = data_gt(:,6);
    q3 = data_gt(:,7);
    q4 = data_gt(:,8);

    quat = [q1 q2 q3 q4];
end