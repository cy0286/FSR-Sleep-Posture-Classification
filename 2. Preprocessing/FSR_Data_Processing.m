% Set up the environment
clc; close all; clear all;

sub = '1_SCY';
dir_name = ['C:\Users\송채영\Desktop\HAI-lab\Research Data\[2401~2402] UWB_Biopac_Bed dataset\', sprintf('%s',sub)];

Final_SyncData.Fs_pvdf = 250; % Sampling rate for PVDF
Final_SyncData.Fs_fsr = 10;   % Sampling rate for FSR

cd(sprintf('%s/PVDF_FSR',dir_name))

% Initialize data size arrays
pvdf_size = [];
fsr_ch_size = [];
fsr_size = [];

% Loop through each device to process data
for device_num = 1 : 4
    current_dir = dir([sprintf('%s', dir_name), '\PVDF_FSR']);

    % ------------------- PVDF Data Processing -------------------
    % Find index for pvdf_raw files
    idx = find(~cellfun(@isempty, strfind({current_dir.name}, 'pvdf_raw')));
    fname = current_dir(idx(device_num)).name;
    fid = fopen(fname, "r");
    pvdf = fread(fid, Inf, "int32");
    fclose(fid);
    Final_SyncData.pvdf_all{1,device_num} = pvdf;
    pvdf_size(device_num) = size(pvdf,1);

    % ------------------- FSR Data Processing --------------------
    % Find index for fsr70_raw files
    idx = find(~cellfun(@isempty, strfind({current_dir.name}, 'fsr70_raw')));
    fname = current_dir(idx(device_num)).name;
    fid = fopen(fname, "r");
    data = fread(fid, Inf, "uint16");
    fclose(fid);
    
    % Resample FSR data into 7 channels
    tmp_fsr = reshape(data', [7, length(data)/7]);

    % ------------------- 1-second window -------------------
    fsr = zeros(7, floor(length(data)/7/10));
    for sec = 1 : floor(length(data)/7/10)
        % Calculate mean over 10 samples (1-second window at 10 Hz)
        fsr(:, sec) = mean(tmp_fsr(:, (sec-1)*10+1 : sec*10), 2);
    end
    Final_SyncData.fsr_all{1, device_num} = fsr;

    % ------------------- 5-second window -------------------
    fsr_5sec = zeros(7, floor(length(data)/7/50));
    for sec = 1 : floor(length(data)/7/50)
        % Calculate mean over 50 samples (5-second window at 10 Hz)
        fsr_5sec(:, sec) = mean(tmp_fsr(:, (sec-1)*50+1 : sec*50), 2);
    end
    Final_SyncData.fsr_5sec{1, device_num} = fsr_5sec;

    % ------------------- 10-second window -------------------
    fsr_10sec = zeros(7, floor(length(data)/7/100));
    for sec = 1 : floor(length(data)/7/100)
        % Calculate mean over 100 samples (10-second window at 10 Hz)
        fsr_10sec(:, sec) = mean(tmp_fsr(:, (sec-1)*100+1 : sec*100), 2);
    end
    Final_SyncData.fsr_10sec{1, device_num} = fsr_10sec; 

    % Save the minimum size for each channel
    for fsr_ch = 1 : 7
        fsr_size(fsr_ch) = size(fsr(fsr_ch,:), 2);
    end
    fsr_ch_size(device_num) = min(fsr_size);
end

% ------------------- Synchronize Data Sizes --------------------
min_pvdf_size = min(pvdf_size);
min_fsr_size = min(fsr_ch_size);
for device_num = 1 : 4
    if size(Final_SyncData.pvdf_all{1, device_num}, 1) > min_pvdf_size
        diff_size = size(Final_SyncData.pvdf_all{1, device_num}, 1) - min_pvdf_size;
        Final_SyncData.pvdf_all{1, device_num} = Final_SyncData.pvdf_all{1, device_num}(diff_size+1 : end);

        diff_size = size(Final_SyncData.fsr_all{1, device_num}, 2) - min_fsr_size;
        Final_SyncData.fsr_all{1, device_num} = Final_SyncData.fsr_all{1, device_num}(:, diff_size+1:end);
    end
end

% ------------------- Reorganize and Label Data --------------------
% Reorganize data for 1-second windows
reorganized_data_1s = zeros(4, 7, 3200);
for i = 1:4
    current_array = Final_SyncData.fsr_all{1, i}; % 1-second window data
    for j = 1:7
        reorganized_data_1s(i, j, 1:800) = current_array(j, 51:850);
        reorganized_data_1s(i, j, 801:1600) = current_array(j, 951:1750);
        reorganized_data_1s(i, j, 1601:2400) = current_array(j, 1851:2650);
        reorganized_data_1s(i, j, 2401:3200) = current_array(j, 2751:3550);
    end
end
labels_1s = zeros(1, 3200); % Labels for 1-second windows
labels_1s(1:800) = 0;
labels_1s(801:1600) = 1;
labels_1s(1601:2400) = 2;
labels_1s(2401:3200) = 3;

% Reorganize data for 5-second windows
reorganized_data_5s = zeros(4, 7, 640); % 3200 / 5 = 640
for i = 1:4
    current_array = Final_SyncData.fsr_5sec{1, i}; % 5-second window data
    for j = 1:7
        reorganized_data_5s(i, j, 1:160) = current_array(j, 11:170); % Scaled (51~850)/5
        reorganized_data_5s(i, j, 161:320) = current_array(j, 191:350);
        reorganized_data_5s(i, j, 321:480) = current_array(j, 371:530);
        reorganized_data_5s(i, j, 481:640) = current_array(j, 551:710);
    end
end
labels_5s = zeros(1, 640); % Labels for 5-second windows
labels_5s(1:160) = 0;
labels_5s(161:320) = 1;
labels_5s(321:480) = 2;
labels_5s(481:640) = 3;

% Reorganize data for 10-second windows
reorganized_data_10s = zeros(4, 7, 320); % 3200 / 10 = 320
for i = 1:4
    current_array = Final_SyncData.fsr_10sec{1, i}; % 10-second window data
    for j = 1:7
        reorganized_data_10s(i, j, 1:80) = current_array(j, 6:85); % Scaled (51~850)/10
        reorganized_data_10s(i, j, 81:160) = current_array(j, 96:175);
        reorganized_data_10s(i, j, 161:240) = current_array(j, 186:265);
        reorganized_data_10s(i, j, 241:320) = current_array(j, 276:355);
    end
end
labels_10s = zeros(1, 320); % Labels for 10-second windows
labels_10s(1:80) = 0;
labels_10s(81:160) = 1;
labels_10s(161:240) = 2;
labels_10s(241:320) = 3;

% ------------------- Save Data --------------------
save_dir = 'C:\Users\송채영\Desktop\송채영\HAI\code\data\1\';

% Save reorganized data and labels for each window size
save([save_dir 'fsr_1s.mat'], 'reorganized_data_1s', 'labels_1s');
save([save_dir 'fsr_5s.mat'], 'reorganized_data_5s', 'labels_5s');
save([save_dir 'fsr_10s.mat'], 'reorganized_data_10s', 'labels_10s');