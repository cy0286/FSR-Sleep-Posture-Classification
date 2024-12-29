import numpy as np
import scipy.io

for j in range(1,53):
    file_path = f"C:/Users/송채영/Desktop/송채영/HAI/code/data/{j}/fsr.mat"

    # Load the .mat file
    data = scipy.io.loadmat(file_path, struct_as_record=False, squeeze_me=True)
    
    # Check the labels structure in the file
    if 'labels' in data:
        labels_struct = data['labels']
    
        if isinstance(labels_struct, np.ndarray):
            labels_data = labels_struct
        else:
            print("No valid label data found in 'labels' structure.")
    else:
        print("Data structure 'labels' not found in the file.")

    # Check the reorganized_data structure in the file
    if 'reorganized_data' in data:
        reorganized_data_struct = data['reorganized_data']
    
        if isinstance(reorganized_data_struct, np.ndarray):
            reorganized_data = reorganized_data_struct
        else:
            print("No valid label data found in 'reorganized_data' structure.")
    else:
        print("Data structure 'reorganized_data' not found in the file.")

    # Reshape reorganized_data and transpose it
    reorganized_data = reorganized_data.reshape(28, 3200)
    reorganized_data = np.transpose(reorganized_data)

    # Combine transposed data with labels and save as .npy
    combined_data = np.column_stack((reorganized_data, labels_data))
    np.save(f"C:/Users/송채영/Desktop/송채영/HAI/code/model/{j}.npy",combined_data)