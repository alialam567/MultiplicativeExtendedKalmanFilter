%% Multiplicative Extended Kalman Filter (MEKF) for Drone Localization

% Initialization
clc; clear; close all;
f = 100; % Frequency
dt = 1/f; % Time step (100 Hz)
duration = 10; % in seconds
N = duration * f; % Number of time steps

% State vector: [position; velocity; quaternion; gyro_bias; accel_bias]
x = [zeros(3,1); zeros(3,1); [1; 0; 0; 0]; zeros(3,1); zeros(3,1)];

% Covariance matrices
P = eye(15) * 0.1; % Initial covariance
Q = eye(15) * 0.01; % Process noise covariance
R_pos = eye(3) * 0.1;  % Position noise
R_ori = eye(4) * 0.01; % Orientation noise
R = blkdiag(R_pos, R_ori); % Combine into a block diagonal matrix

% Generate simulated IMU and ToF data
imu_data = randn(6, N) * 0.01; % [Gyro; Accel]
tof_data = cumsum(randn(3, N) * 0.01, 2); % Simulated position data

% Storage for results
state_hist = zeros(15, N);

%% Main Loop
for k = 1:N
    % Prediction Step
    [x_pred, P_pred] = predictState(x, P, dt, Q, imu_data(:,k));
    
    % Measurement Update
    z = tof_data(:,k);
    H = [eye(3), zeros(3, 3), zeros(3, 4), zeros(3, 3), zeros(3, 3);
     zeros(4, 3), zeros(4, 3), eye(4), zeros(4, 3), zeros(4, 3)];
    % Measurement matrix for position only

    [x, P] = updateState(x_pred, P_pred, R, z, H);

    % Normalize quaternion
    x(7:10) = x(7:10) / norm(x(7:10));

    % Store state
    state_hist(:,k) = x;
end

%% Functions
function [x_pred, P_pred] = predictState(x, P, dt, Q, imu_data)
    % Extract states
    p = x(1:3);
    v = x(4:6);
    q = x(7:10);
    b_g = x(11:13);
    b_a = x(14:16);

    % IMU data
    w = imu_data(1:3) - b_g;
    a = imu_data(4:6) - b_a;
    g = [0; 0; -9.81];

    % Quaternion kinematics using small-angle approximation
    Omega_w = OmegaOp(w);
    dq = (1/2) * Omega_w * q;
    
    % State propagation
    p_next = p + v * dt;
    v_next = v + ((rot_mat_from_quat(q)*a) + g) * dt;
    q_next = q + dq * dt;
    q_next = q_next / norm(q_next);

    % Update state vector
    x_pred = [p_next; v_next; q_next; b_g; b_a];

    % State transition matrix (F) approximation
    F = eye(15);
    F(1:3, 4:6) = eye(3) * dt;
    F(4:6, 7:9) = -(rot_mat_from_quat(q)*a) * dt;
    F(4:6, 14:16) = -(rot_mat_from_quat(q)*eye(3)) * dt;
    F(7:9, 7:9) = -skewSymmetric(omega - b_g) * dt;
    F(7:9, 11:13) = -eye(3) * dt;

    % Covariance propagation
    P_pred = F * P * F' + Q;
end

function [x, P] = updateState(x_pred, P_pred, R, z, H)
    % Measurement Update
    K = P_pred * H' / (H * P_pred * H' + R);
    delta_x = K * (z - H * x_pred);

    % Apply error state correction using the MEKF method
    x = x_pred;
    x(1:3) = x(1:3) + delta_x(1:3);
    x(4:6) = x(4:6) + delta_x(4:6);
    delta_phi = delta_x(7:9);
    delta_q = [1; delta_phi / 2];
    x(7:10) = quatmultiply(delta_q, x_pred(7:10));
    x(11:16) = x_pred(11:16) + delta_x(10:15);

    % Covariance update
    P = (eye(size(K,1)) - K * H) * P_pred;
end


% maybe fix F (state transition matrix)
% maybe fix Q (update with the correct sensor shit)
