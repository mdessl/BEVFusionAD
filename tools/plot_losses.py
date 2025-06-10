import json
import matplotlib.pyplot as plt
import numpy as np
import re

# Function to read the log file and extract data
def parse_log_file(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    data = []
    for line in lines:
        line = line.strip()
        if line and not line.startswith('...'):
            try:
                entry = json.loads(line)
                if entry.get('mode') == 'train':
                    data.append(entry)
            except json.JSONDecodeError:
                continue
    
    return data

# Extract data from log file
log_file = "work_dirs/0203_sbnet_first/20250301_222858.log.json"
data = parse_log_file(log_file)

# Extract iterations and loss values
# Use a continuous iteration counter instead of per-epoch iterations
iterations = [i for i in range(len(data))]
total_loss = [entry['loss'] for entry in data]
loss_heatmap = [entry['loss_heatmap'] for entry in data]
loss_cls = [entry['layer_-1_loss_cls'] for entry in data]
loss_bbox = [entry['layer_-1_loss_bbox'] for entry in data]
matched_ious = [entry['matched_ious'] for entry in data]

# Create the plot
plt.figure(figsize=(12, 10))

# Plot total loss
plt.subplot(3, 2, 1)
plt.plot(iterations, total_loss, 'b-', label='Total Loss')
plt.title('Total Loss')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.grid(True, alpha=0.3)

# Plot heatmap loss
plt.subplot(3, 2, 2)
plt.plot(iterations, loss_heatmap, 'r-', label='Heatmap Loss')
plt.title('Heatmap Loss')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.grid(True, alpha=0.3)

# Plot classification loss
plt.subplot(3, 2, 3)
plt.plot(iterations, loss_cls, 'g-', label='Classification Loss')
plt.title('Classification Loss')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.grid(True, alpha=0.3)

# Plot bounding box loss
plt.subplot(3, 2, 4)
plt.plot(iterations, loss_bbox, 'm-', label='Bounding Box Loss')
plt.title('Bounding Box Loss')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.grid(True, alpha=0.3)

# Plot matched IoUs
plt.subplot(3, 2, 5)
plt.plot(iterations, matched_ious, 'c-', label='Matched IoUs')
plt.title('Matched IoUs')
plt.xlabel('Iteration')
plt.ylabel('IoU')
plt.grid(True, alpha=0.3)

# Plot all losses together for comparison
plt.subplot(3, 2, 6)
plt.plot(iterations, total_loss, 'b-', label='Total Loss')
plt.plot(iterations, loss_heatmap, 'r-', label='Heatmap Loss')
plt.plot(iterations, loss_cls, 'g-', label='Classification Loss')
plt.plot(iterations, loss_bbox, 'm-', label='Bounding Box Loss')
plt.title('All Losses')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_losses.png', dpi=300)
plt.show()

# Calculate and print statistics
print("Loss Statistics:")
print(f"Total Loss - Min: {min(total_loss):.4f}, Max: {max(total_loss):.4f}, Mean: {np.mean(total_loss):.4f}")
print(f"Heatmap Loss - Min: {min(loss_heatmap):.4f}, Max: {max(loss_heatmap):.4f}, Mean: {np.mean(loss_heatmap):.4f}")
print(f"Classification Loss - Min: {min(loss_cls):.4f}, Max: {max(loss_cls):.4f}, Mean: {np.mean(loss_cls):.4f}")
print(f"Bounding Box Loss - Min: {min(loss_bbox):.4f}, Max: {max(loss_bbox):.4f}, Mean: {np.mean(loss_bbox):.4f}")
print(f"Matched IoUs - Min: {min(matched_ious):.4f}, Max: {max(matched_ious):.4f}, Mean: {np.mean(matched_ious):.4f}")

# Calculate moving averages to see trends more clearly
window_size = 20
total_loss_ma = np.convolve(total_loss, np.ones(window_size)/window_size, mode='valid')
iterations_ma = iterations[window_size-1:]

plt.figure(figsize=(10, 6))
plt.plot(iterations, total_loss, 'b-', alpha=0.3, label='Total Loss')
plt.plot(iterations_ma, total_loss_ma, 'r-', label=f'Moving Average (window={window_size})')
plt.title('Total Loss with Moving Average')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('total_loss_moving_average.png', dpi=300)
plt.show()