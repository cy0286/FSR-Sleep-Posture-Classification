clc; close all; clear all;

sub = '1_SCY';
dir_name = ['C:\Users\송채영\Desktop\HAI-lab\Research Data\[2401~2402] UWB_Biopac_Bed dataset\', sprintf('%s',sub)];

Final_SyncData.Fs_pvdf = 250;
Final_SyncData.Fs_fsr = 10;

cd(sprintf('%s/PVDF_FSR',dir_name))

pvdf_size = [];
fsr_ch_size = [];
fsr_size = [];

for device_num = 1 : 4
    current_dir = dir([sprintf('%s', dir_name), '\PVDF_FSR']);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%% PVDF %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    idx = find(~cellfun(@isempty, strfind({current_dir.name}, 'pvdf_raw')));
    fname = current_dir(idx(device_num)).name;
    fid = fopen(fname, "r");
    pvdf = fread(fid, Inf, "int32");
    fclose(fid);
    Final_SyncData.pvdf_all{1,device_num} = pvdf;
    pvdf_size(device_num) = size(pvdf,1);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%% FSR %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    idx = find(~cellfun(@isempty, strfind({current_dir.name}, 'fsr70_raw')));
    fname = current_dir(idx(device_num)).name;
    fid = fopen(fname, "r");
    data = fread(fid, Inf, "uint16");
    fclose(fid);

    tmp_fsr = reshape(data', [7, length(data)/7]);
    fsr = zeros(7, floor(length(data)/7/10));
    for sec = 1 : floor(length(data)/7/10)
        fsr(:, sec) = mean(tmp_fsr(:, (sec-1)*10+1 : sec*10), 2);
    end
    Final_SyncData.fsr_all{1,device_num} = fsr;
    
    for fsr_ch = 1 : 7
        fsr_size(fsr_ch) = size(fsr(fsr_ch,:), 2);
    end
    fsr_ch_size(device_num) = min(fsr_size);
end

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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

reorganized_data = zeros(4, 7, 3200);
labels = zeros(1, 3200);

for i = 1:4
    current_array = Final_SyncData.fsr_all{1, i};
    
    for j = 1:7
        reorganized_data(i, j, 1:800) = current_array(j, 51:850);
        reorganized_data(i, j, 801:1600) = current_array(j, 951:1750);
        reorganized_data(i, j, 1601:2400) = current_array(j, 1851:2650);
        reorganized_data(i, j, 2401:3200) = current_array(j, 2751:3550);
    end
end

labels(1:800) = 0;
labels(801:1600) = 1;
labels(1601:2400) = 2;
labels(2401:3200) = 3;

save_dir = 'C:\Users\송채영\Desktop\송채영\HAI\code\data\1\'; 
save([save_dir 'fsr.mat'], 'reorganized_data', 'labels');

