clc; clear; close all;
rng(42, "twister");  % Reproducible random training and simulation data

%% ================= Simulation Parameters =================
N = 1e4;
EbN0_dB = 0:5:30;
num_iter = 3;
sigma_R = 0.3;

ser_ook = zeros(size(EbN0_dB));
ser_qam = zeros(size(EbN0_dB));
ser_pam = zeros(size(EbN0_dB));

%% ================= MLP Training Parameters =================
N_train = 3000;
h_train = exp(sqrt(2)*sigma_R*randn(1,N_train) - sigma_R^2);
n_train = randn(1,N_train)/sqrt(2);

%% ================= OOK Training =================
ip_ook = randi([0 1], 1, N_train);
y_train_ook = h_train .* ip_ook + 10^(-15/20)*n_train;

X_ook = real(y_train_ook)' ./ std(real(y_train_ook));
Y_ook = ip_ook';

mlp_ook = fitcnet(X_ook, Y_ook, ...
    'LayerSizes', 10, 'Activations', 'relu', ...
    'Standardize', false, 'IterationLimit', 50);

%% ================= 4-PAM Training =================
pam_symbols = [-3 -1 1 3];
ip_pam_idx = randi([1 4], 1, N_train);
ip_pam = pam_symbols(ip_pam_idx);

y_train_pam = h_train .* ip_pam + 10^(-15/20)*n_train;
X_pam = real(y_train_pam)' ./ std(real(y_train_pam));
Y_pam = ip_pam_idx';

mlp_pam = fitcnet(X_pam, Y_pam, ...
    'LayerSizes', 10, 'Activations', 'relu', ...
    'Standardize', false, 'IterationLimit', 50);

%% ================= 4-QAM Training =================
qam_symbols = [1+1i, 1-1i, -1+1i, -1-1i];
ip_qam_idx = randi([1 4], 1, N_train);
ip_qam = qam_symbols(ip_qam_idx);

y_train_qam = h_train .* ip_qam + 10^(-15/20)*n_train;
X_qam = [real(y_train_qam)', imag(y_train_qam)'];
Y_qam = ip_qam_idx';

mlp_qam = fitcnet(X_qam, Y_qam, ...
    'LayerSizes', 10, 'Activations', 'relu', ...
    'Standardize', false, 'IterationLimit', 50);

%% ================= Simulation Over SNR =================
for idx = 1:length(EbN0_dB)
    snr = EbN0_dB(idx);
    fprintf("SNR = %d dB\n", snr);

    err_ook = 0; err_pam = 0; err_qam = 0;

    for iter = 1:num_iter
        h = exp(sqrt(2)*sigma_R*randn(1,N) - sigma_R^2);
        n = randn(1,N)/sqrt(2);

        % ----- OOK -----
        ip = randi([0 1], 1, N);
        y = h .* ip + 10^(-snr/20)*n;
        X = real(y)' ./ std(real(y));
        y_pred = predict(mlp_ook, X)';
        err_ook = err_ook + sum(ip ~= y_pred);

        % ----- 4-PAM -----
        ip_idx = randi([1 4], 1, N);
        ip_pam = pam_symbols(ip_idx);
        y_pam = h .* ip_pam + 10^(-snr/20)*n;
        X_pam = real(y_pam)' ./ std(real(y_pam));
        y_pred_pam = predict(mlp_pam, X_pam)';
        err_pam = err_pam + sum(ip_idx ~= y_pred_pam);

        % ----- 4-QAM -----
        ip_qam_idx = randi([1 4], 1, N);
        ip_qam = qam_symbols(ip_qam_idx);
        y_qam = h .* ip_qam + 10^(-snr/20)*n;
        X_qam = [real(y_qam)', imag(y_qam)'];
        y_pred_qam = predict(mlp_qam, X_qam)';
        err_qam = err_qam + sum(ip_qam_idx ~= y_pred_qam);
    end

    ser_ook(idx) = err_ook / (N * num_iter);
    ser_pam(idx) = err_pam / (N * num_iter);
    ser_qam(idx) = err_qam / (N * num_iter);
end

%% ================= Plot 1: MLP-based BER Comparison =================
figure;
semilogy(EbN0_dB, ser_ook, '-o', 'LineWidth', 1.5); hold on;
semilogy(EbN0_dB, ser_pam, '-s', 'LineWidth', 1.5);
semilogy(EbN0_dB, ser_qam, '-d', 'LineWidth', 1.5);
grid on;
xlabel('Eb/N0 (dB)');
ylabel('Symbol Error Rate (SER)');
legend('MLP + OOK', 'MLP + 4-PAM', 'MLP + 4-QAM');
title('MLP Symbol Detection under Log-Normal FSO Turbulence');
ylim([1e-4 1]);

%% ================= Plot 2: Theoretical BER for QAM =================
SNR = 10.^(EbN0_dB/10);
ber_4qam  = qfunc(sqrt(2*SNR));
ber_16qam = 3/8 * qfunc(sqrt((4/5)*SNR));
ber_64qam = 7/12 * qfunc(sqrt((1/7)*SNR));

figure;
semilogy(EbN0_dB, ber_4qam, '-o', 'LineWidth', 2); hold on;
semilogy(EbN0_dB, ber_16qam, '-s', 'LineWidth', 2);
semilogy(EbN0_dB, ber_64qam, '-^', 'LineWidth', 2);
grid on;
xlabel('SNR (dB)');
ylabel('BER');
legend('4-QAM', '16-QAM', '64-QAM');
title('Theoretical BER of QAM Schemes');

%% ================= Plot 3: 64-QAM under Turbulence Levels =================
ber_weak     = 7/12 * qfunc(sqrt((1/7)*SNR));
ber_moderate = 7/12 * qfunc(sqrt((1/9)*SNR));
ber_strong   = 7/12 * qfunc(sqrt((1/12)*SNR));

figure;
semilogy(EbN0_dB, ber_weak, '-o', 'LineWidth', 2); hold on;
semilogy(EbN0_dB, ber_moderate, '-s', 'LineWidth', 2);
semilogy(EbN0_dB, ber_strong, '-^', 'LineWidth', 2);
grid on;
xlabel('SNR (dB)');
ylabel('BER');
legend('Weak', 'Moderate', 'Strong');
title('BER vs SNR for 64-QAM under Turbulence');

%% ================= Plot 4: Confusion Matrix (ESSENTIAL) =================
snr_cm = 15;  % Representative SNR

h = exp(sqrt(2)*sigma_R*randn(1,N) - sigma_R^2);
n = randn(1,N)/sqrt(2);

ip_qam_idx = randi([1 4], 1, N);
ip_qam = qam_symbols(ip_qam_idx);

y_qam = h .* ip_qam + 10^(-snr_cm/20)*n;
X_qam = [real(y_qam)', imag(y_qam)'];

y_pred_qam = predict(mlp_qam, X_qam)';

figure;
confusionchart(ip_qam_idx', y_pred_qam', ...
    'RowSummary','row-normalized', ...
    'ColumnSummary','column-normalized');
title('Confusion Matrix: MLP-based 4-QAM Detection (SNR = 15 dB)');



