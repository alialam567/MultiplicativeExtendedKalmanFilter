classdef MEKF_TOF < handle
    properties
        estimate % quaternion as 1x4 [w x y z]
        estimate_covariance
        gyro_bias
        accelerometer_bias

        position
        velocity

        observation_covariance
        G
        K
        H

        gyro_cov_mat
        gyro_bias_cov_mat
        accel_cov_mat
        accel_bias_cov_mat
        tof_cov_mat
        tof_bias_cov_mat

        h_predicted_log
        acc_meas_norm_log
        delta_theta_log
        velocity_log
        position_log
    end

    methods
        function obj = MEKF_TOF(initial_est, initial_pos, initial_vel,estimate_covariance, ...
                gyro_cov, gyro_bias_cov, accel_proc_cov, ...
                accel_bias_cov, accel_obs_cov, tof_proc_cov, tof_bias_cov, tof_obs_cov)
            obj.estimate = initial_est; % [w x y z]
            obj.estimate_covariance = estimate_covariance * eye(18);

            obj.H = zeros(6, 18);

            obj.observation_covariance(1:3,1:3) = accel_obs_cov * eye(3);
            obj.observation_covariance(4:6,4:6) = tof_obs_cov * eye(3);
            obj.gyro_bias = zeros(3, 1);
            obj.accelerometer_bias = zeros(3, 1);
            obj.velocity = initial_vel';
            obj.position = initial_pos';

            obj.G = zeros(18);
            obj.G(1:3, 10:12) = -eye(3);
            obj.G(7:9, 4:6) = eye(3);

            obj.gyro_cov_mat = gyro_cov * eye(3);
            obj.gyro_bias_cov_mat = gyro_bias_cov * eye(3);
            obj.accel_cov_mat = accel_proc_cov * eye(3);
            obj.accel_bias_cov_mat = accel_bias_cov * eye(3);
            obj.tof_cov_mat = tof_proc_cov * eye(3);
            obj.tof_bias_cov_mat = tof_bias_cov * eye(3);

            obj.h_predicted_log = [];
            obj.acc_meas_norm_log = [];
            obj.delta_theta_log = [];
            obj.velocity_log = [];
            obj.position_log = [];
        end

        function Q = process_covariance(obj, dt)
            Q = zeros(18);
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
            Q(16:18,16:18) = obj.tof_bias_cov_mat*dt;
        end

        function predict(obj, gyro_meas, acc_meas, dt)
            
            gyro_meas = gyro_meas - obj.gyro_bias;
            
            acc_meas = acc_meas - obj.accelerometer_bias;
            acc_meas_norm = acc_meas / norm(acc_meas);
        
            % Quaternion integration
            if norm(gyro_meas) > 1e-8
                omega_quat = [0, gyro_meas(1), gyro_meas(2), gyro_meas(3)];
                q_dot = 0.5 * quatmultiply(obj.estimate, omega_quat);
                obj.estimate = obj.estimate + dt * q_dot;
                obj.estimate = obj.estimate / norm(obj.estimate);
            end
        
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
            
            F = eye(18) + obj.G * dt;
        
            % Predict covariance
            obj.estimate_covariance = F * obj.estimate_covariance * F' + obj.process_covariance(dt);
        
            % Kalman gain
            obj.H = zeros(6, 18);
            h_predicted = quatrotate(quatinv(obj.estimate), [0 0 -1])'; % Transposed to 3x1
            obj.H(1:3, 1:3) = -skewSymmetric(h_predicted);
            obj.H(4:6, 7:9) = eye(3);
            PH_T = obj.estimate_covariance * obj.H';
            S = obj.H * PH_T + obj.observation_covariance;
            obj.K = PH_T / S;
        end

        function update(obj, acc_meas, pf_meas)

            acc_meas = acc_meas - obj.accelerometer_bias;
            acc_meas_norm = acc_meas / norm(acc_meas);

            observation = zeros(6,1);
            observation(1:3) = acc_meas_norm;
            observation(4:6) = pf_meas(1:3);
            predicted_observation = zeros(6,1);
            predicted_observation(1:3) = quatrotate(quatinv(obj.estimate), [0 0 -1])';
            predicted_observation(4:6) = obj.position;

            aposteriori_state = obj. K * (observation - predicted_observation);

            % Quaternion correction
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
            obj.estimate_covariance = (eye(18) - obj.K * obj.H) * obj.estimate_covariance;

            % Logging
            obj.h_predicted_log(:, end+1) = quatrotate(quatinv(obj.estimate), [0 0 -1])';
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
