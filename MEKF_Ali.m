classdef MEKF_Ali < handle
    properties
        estimate % quaternion as 1x4 [w x y z]
        estimate_covariance
        gyro_bias
        accelerometer_bias

        position
        velocity

        observation_covariance
        G

        gyro_cov_mat
        gyro_bias_cov_mat
        accel_cov_mat
        accel_bias_cov_mat

        h_predicted_log
        acc_meas_norm_log
        delta_theta_log
        velocity_log
        position_log
    end

    methods
        function obj = MEKF_Ali(initial_est, estimate_covariance, ...
                gyro_cov, gyro_bias_cov, accel_proc_cov, ...
                accel_bias_cov, accel_obs_cov)
            obj.estimate = initial_est; % [w x y z]
            obj.estimate_covariance = estimate_covariance * eye(15);

            obj.observation_covariance(1:3,1:3) = accel_obs_cov * eye(3);
            obj.gyro_bias = zeros(3, 1);
            obj.accelerometer_bias = zeros(3, 1);
            obj.velocity = zeros(3, 1);
            obj.position = zeros(3, 1);

            obj.G = zeros(15);
            obj.G(1:3, 10:12) = -eye(3);
            obj.G(7:9, 4:6) = eye(3);

            obj.gyro_cov_mat = gyro_cov * eye(3);
            obj.gyro_bias_cov_mat = gyro_bias_cov * eye(3);
            obj.accel_cov_mat = accel_proc_cov * eye(3);
            obj.accel_bias_cov_mat = accel_bias_cov * eye(3);

            obj.h_predicted_log = [];
            obj.acc_meas_norm_log = [];
            obj.delta_theta_log = [];
            obj.velocity_log = [];
            obj.position_log = [];
        end

        function Q = process_covariance(obj, dt)
            Q = zeros(15);
            Q(1:3,1:3) = obj.gyro_cov_mat*dt + obj.gyro_bias_cov_mat*(dt^3)/3;
            Q(1:3,10:12) = -obj.gyro_bias_cov_mat*(dt^2)/2;
            Q(4:6,4:6) = obj.accel_cov_mat*dt + obj.accel_bias_cov_mat*(dt^3)/3;
            Q(4:6,7:9) = obj.accel_bias_cov_mat*(dt^4)/8 + obj.accel_cov_mat*(dt^2)/2;
            Q(4:6,13:15) = -obj.accel_bias_cov_mat*(dt^2)/2;
            Q(7:9,4:6) = obj.accel_cov_mat*(dt^2)/2 + obj.accel_bias_cov_mat*(dt^4)/8;
            Q(7:9,7:9) = obj.accel_cov_mat*(dt^3)/3 + obj.accel_bias_cov_mat*(dt^5)/20;
            Q(7:9,13:15) = -obj.accel_bias_cov_mat*(dt^3)/6;
            Q(10:12,1:3) = -obj.gyro_bias_cov_mat*(dt^2)/2;
            Q(10:12,10:12) = obj.gyro_bias_cov_mat*dt;
            Q(13:15,4:6) = -obj.accel_bias_cov_mat*(dt^2)/2;
            Q(13:15,7:9) = -obj.accel_bias_cov_mat*(dt^3)/6;
            Q(13:15,13:15) = obj.accel_bias_cov_mat*dt;
        end

        function update(obj, gyro_meas, acc_meas, dt)
            gyro_meas = gyro_meas - obj.gyro_bias;
            
            acc_meas = acc_meas - obj.accelerometer_bias;
            acc_meas_norm = acc_meas / norm(acc_meas);

            % Quaternion integration
            omega_quat = [0, gyro_meas(1), gyro_meas(2), gyro_meas(3)]; % Row vector [w x y z]
            q_dot = 0.5 * quatmultiply(obj.estimate, omega_quat); % Right multiplication (q * ω)
            obj.estimate = obj.estimate + dt * q_dot;
            obj.estimate = obj.estimate / norm(obj.estimate);
            %obj.estimate
        
            % Rotation matrix
            R = quat2dcm(obj.estimate);

            % Update velocity and position
            acc_world = R * acc_meas - (quatrotate(quatinv(obj.estimate), [0 0 -1])')*9.71;
            obj.velocity = obj.velocity + acc_world * dt;
            obj.position = obj.position + obj.velocity * dt;

            % Process model Jacobian
            obj.G(1:3, 1:3) = -skewSymmetric(gyro_meas); % d(theta_dot)/d(theta)
            obj.G(4:6, 1:3) = -R * skewSymmetric(acc_meas_norm); % d(v_dot)/d(theta)
            obj.G(4:6, 13:15) = -R; % d(v_dot)/d(ba)
            
            F = eye(15) + obj.G * dt;
        
            % Predict covariance
            obj.estimate_covariance = F * obj.estimate_covariance * F' + obj.process_covariance(dt);
        
            % Kalman gain
            H = zeros(3, 15);
            h_predicted = quatrotate(quatinv(obj.estimate), [0 0 -1])'; % Transposed to 3x1
            H(1:3, 1:3) = -skewSymmetric(h_predicted);
            PH_T = obj.estimate_covariance * H';
            S = H * PH_T + obj.observation_covariance;
            K = PH_T / S;
            
            error = ((acc_meas_norm) - h_predicted); % Both 3x1
            aposteriori_state = K * error;
            
            % Quaternion update (column vector)
            delta_theta = aposteriori_state(1:3);
            dq = [1, 0.5 * delta_theta(1), 0.5 * delta_theta(2), 0.5 * delta_theta(3)];
            obj.estimate = quatmultiply(obj.estimate, dq); 
            obj.estimate = obj.estimate / norm(obj.estimate);

            % Velocity and position correction
            obj.velocity = obj.velocity + aposteriori_state(4:6);
            obj.position = obj.position + aposteriori_state(7:9);
        
            % Bias updates (direct column addition)
            obj.gyro_bias = obj.gyro_bias + aposteriori_state(10:12);
            obj.accelerometer_bias = obj.accelerometer_bias + aposteriori_state(13:15);
        
            % Covariance update
            obj.estimate_covariance = (eye(15) - K * H) * obj.estimate_covariance;


            % Logging
            obj.h_predicted_log(:, end+1) = h_predicted;
            obj.acc_meas_norm_log(:, end+1) = acc_meas_norm;
            obj.delta_theta_log(:, end+1) = delta_theta;
            obj.velocity_log(:, end+1) = obj.velocity;
            obj.position_log(:, end+1) = obj.position;
        end
    end
end

function S = skewSymmetric(v)
    S = [  0   -v(3)  v(2);
          v(3)   0   -v(1);
         -v(2) v(1)    0 ];
end
