import numpy as np
import matplotlib.pyplot as plt

# Define window sizes and number of layers
window_sizes = [1, 5, 10]
num_layers = [1, 2, 3]

# Define accuracy results for two classes (class 3 and class 4)
results1 = {
    (1, 1): 0.8408, (1, 2): 0.8504, (1, 3): 0.8358,
    (5, 1): 0.8534, (5, 2): 0.8519, (5, 3): 0.8566,
    (10, 1): 0.8451, (10, 2): 0.8469, (10, 3): 0.8644
}

results2 = {
    (1, 1): 0.7971, (1, 2): 0.7949, (1, 3): 0.8003,
    (5, 1): 0.7872, (5, 2): 0.7961, (5, 3): 0.8171,
    (10, 1): 0.8003, (10, 2): 0.8126, (10, 3): 0.82
}

# Initialize accuracy data arrays for each class
accuracy_data1 = np.zeros((len(window_sizes), len(num_layers)))
accuracy_data2 = np.zeros((len(window_sizes), len(num_layers)))

# Fill accuracy data for class 3
for i, window_size in enumerate(window_sizes):
    for j, layers in enumerate(num_layers):
        accuracy_data1[i, j] = results1.get((window_size, layers), 0)

# Fill accuracy data for class 4
for i, window_size in enumerate(window_sizes):
    for j, layers in enumerate(num_layers):
        accuracy_data2[i, j] = results2.get((window_size, layers), 0)

# Create subplots to visualize results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6)) 

# Plot accuracy for class 3
for i, window_size in enumerate(window_sizes):
    ax1.plot(num_layers, accuracy_data1[i], marker='o', label=f'Window Size: {window_size}')
ax1.set_xticks(num_layers)
ax1.set_xlabel('Number of CNN Layers')
ax1.set_ylabel('Accuracy')
ax1.set_title('CNN Accuracy with Window Size and Number of Layers (class 3)')
ax1.legend()
ax1.grid()
ax1.set_ylim(0.75, 0.90) 

# Plot accuracy for class 4
for i, window_size in enumerate(window_sizes):
    ax2.plot(num_layers, accuracy_data2[i], marker='o', label=f'Window Size: {window_size}')
ax2.set_xticks(num_layers)
ax2.set_xlabel('Number of CNN Layers')
ax2.set_ylabel('Accuracy')
ax2.set_title('CNN Accuracy with Window Size and Number of Layers (class 4)')
ax2.legend()
ax2.grid()
ax2.set_ylim(0.75, 0.90) 

plt.tight_layout() 
plt.show()
