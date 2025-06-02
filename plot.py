import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, ConstantInputWarning
import warnings


def calculate_single_channel_shifted_correlation(channel_data_hw, dy, dx):
    """
    Calculates the Pearson correlation coefficient between a single HxW channel and its shifted version by (dy, dx).
    dy, dx: are the offsets of the neighbor relative to the current pixel.
    Compares pixel (y,x) with pixel (y+dy, x+dx).
    Considers only valid overlapping regions.
    """
    H, W = channel_data_hw.shape

    # Determine the valid coordinate range of the original pixel such that both itself and its shifted neighbor are within bounds
    # y_orig_coords: y' such that both y' and y'+dy are valid
    # x_orig_coords: x' such that both x' and x'+dx are valid
    y_orig_coords = np.arange(max(0, -dy), min(H, H - dy))
    x_orig_coords = np.arange(max(0, -dx), min(W, W - dx))

    if len(y_orig_coords) < 1 or len(x_orig_coords) < 1:
        return np.nan # Not enough overlapping region

    original_values_list = []
    neighbor_values_list = []

    for r in y_orig_coords:
        for c in x_orig_coords:
            original_values_list.append(channel_data_hw[r, c])
            neighbor_values_list.append(channel_data_hw[r + dy, c + dx])

    original_vector = np.array(original_values_list)
    neighbor_vector = np.array(neighbor_values_list)

    if len(original_vector) < 2: # Pearson correlation requires at least two points
        return np.nan

    # pearsonr handles constant input (returns NaN and issues a warning)
    corr, _ = pearsonr(original_vector, neighbor_vector)
    return corr

def calculate_average_spatial_autocorrelation_map(tensor_chw, window_size):
    """
    Calculates the global average spatial autocorrelation map.
    For each offset (dy, dx), calculates the average correlation within each channel, then averages across channels.
    """
    C, H, W = tensor_chw.shape
    if window_size % 2 == 0:
        raise ValueError("Window size must be odd.")

    half_window = window_size // 2
    average_correlation_map = np.zeros((window_size, window_size))

    # Ignore pearsonr warnings caused by constant input, and RuntimeWarning when handling NaN
    warnings.filterwarnings("ignore", category=ConstantInputWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")

    for r_idx, dy_offset in enumerate(range(-half_window, half_window + 1)):
        for c_idx, dx_offset in enumerate(range(-half_window, half_window + 1)):
            if dy_offset == 0 and dx_offset == 0:
                average_correlation_map[r_idx, c_idx] = 1.0
            else:
                correlations_for_this_offset_across_channels = []
                for channel_idx in range(C):
                    single_channel_data = tensor_chw[channel_idx, :, :]
                    corr = calculate_single_channel_shifted_correlation(single_channel_data, dy_offset, dx_offset)
                    if not np.isnan(corr):
                        correlations_for_this_offset_across_channels.append(corr)

                if correlations_for_this_offset_across_channels: # If the list is not empty
                    average_correlation_map[r_idx, c_idx] = np.mean(correlations_for_this_offset_across_channels)
                else:
                    average_correlation_map[r_idx, c_idx] = np.nan # If correlation cannot be calculated for any channel

    warnings.resetwarnings() # Reset warning filters
    return average_correlation_map

def calculate_spatial_correlation_from_single_tensor(tensor_chw, center_y, center_x, window_size):
    """
    Calculates the spatial Pearson correlation map from a single (C, H, W) tensor.
    Correlation is calculated between spatial locations, using channels as observation samples.

    Args:
        tensor_chw (np.ndarray): Input tensor, shape (C, H, W).
        center_y (int): y-coordinate of the center point.
        center_x (int): x-coordinate of the center point.
        window_size (int): Neighborhood window size (e.g., 5 for 5x5).

    Returns:
        np.ndarray: Correlation coefficient map of shape (window_size, window_size).
    """
    C, H, W = tensor_chw.shape
    if window_size % 2 == 0:
        raise ValueError("Window size must be odd.")

    half_window = window_size // 2
    correlation_map = np.zeros((window_size, window_size))

    # Values at the center spatial location, across all C channels
    center_values = tensor_chw[:, center_y, center_x] # Shape (C,)

    # Ignore pearsonr warnings caused by constant input
    warnings.filterwarnings("ignore", category=ConstantInputWarning)

    for r_offset_idx, r_offset in enumerate(range(-half_window, half_window + 1)):
        for c_offset_idx, c_offset in enumerate(range(-half_window, half_window + 1)):
            current_y, current_x = center_y + r_offset, center_x + c_offset

            if 0 <= current_y < H and 0 <= current_x < W:
                # Values at the neighboring spatial location, across all C channels
                neighbor_values = tensor_chw[:, current_y, current_x] # Shape (C,)

                correlation, _ = pearsonr(center_values, neighbor_values)

                # If correlation is NaN due to constant input, replace with 0 (or other reasonable value)
                correlation_map[r_offset_idx, c_offset_idx] = correlation if not np.isnan(correlation) else 0.0
            else:
                # If the neighbor is out of bounds, set to NaN
                correlation_map[r_offset_idx, c_offset_idx] = np.nan

    warnings.resetwarnings() # Reset warning filters
    return correlation_map

def plot_correlation_heatmap(correlation_matrix, title="Pearson Correlation Map",
                             vmin_val=None, vmax_val=None): # Allow automatic adjustment of vmin/vmax
    """
    Plots the correlation coefficient heatmap.
    """
    window_size = correlation_matrix.shape[0]
    half_window = window_size // 2
    tick_labels = np.arange(-half_window, half_window + 1)

    plt.figure(figsize=(7, 6)) # Can adjust figure size

    # If the center value is close to 1.0, ensure it's displayed correctly on the heatmap
    # For Pearson correlation, the center point (autocorrelation) should theoretically be 1.0
    # If vmin and vmax are not provided, determine them dynamically from the data (excluding NaN)
    if vmin_val is None:
        vmin_val = np.nanmin(correlation_matrix)
    if vmax_val is None:
        vmax_val = np.nanmax(correlation_matrix)
        # If the center point is 1.0 and much larger than other values, special handling for vmax might be needed
        # Or accept that 1.0 is mapped to the maximum color
        if correlation_matrix[half_window, half_window] == 1.0 and vmax_val < 1.0:
            # If you want to ensure 1.0 is the deepest color and other values have a smaller range
            # Consider setting vmax based on the range of non-center points, as shown in the example plot
            non_center_values = correlation_matrix.copy()
            non_center_values[half_window, half_window] = np.nan # Exclude the center point
            if not np.all(np.isnan(non_center_values)): # Ensure non-center points are not all NaN
                 vmax_val_non_center = np.nanmax(non_center_values)
                 # If the user doesn't specify vmax, and the maximum of non-center points is significantly less than 1
                 # Consider using a vmax similar to the example plot (e.g., 0.04 or 0.1)
                 # Here we use the maximum value of the data itself, and the user can adjust as needed
                 # vmax_val = max(vmax_val, vmax_val_non_center) # Ensure vmax includes at least the non-center values

    current_cmap = plt.cm.get_cmap("Purples").copy() # Get the purple color palette
    current_cmap.set_bad(color='lightgrey') # Set the color for NaN

    sns.heatmap(correlation_matrix, annot=True, fmt=".3f", cmap=current_cmap,
                cbar=True, square=True, xticklabels=tick_labels, yticklabels=tick_labels,
                vmin=vmin_val, vmax=vmax_val,
                annot_kws={"size": 9}, # Adjust annotation text size
                cbar_kws={'label': 'Pearson Correlation'}) # Add label to the color bar

    plt.title(title, fontsize=13)
    plt.xlabel("Relative X position", fontsize=11)
    plt.ylabel("Relative Y position", fontsize=11)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    # plt.savefig(f'./plots/{title}.pdf', dpi=300, format='pdf')
    plt.show()


if __name__ == "__main__":
    # --- Example Usage ---
    # 1. Generate an example tensor (C, H, W) = (320, 32, 32)
    C, H, W = 320, 32, 32
    # To make the correlation more meaningful, we create some structured random data
    # For example, let there be some shared patterns between channels, or some spatial smoothness
    np.random.seed(0)
    example_tensor = np.random.rand(C, H, W)
    # Add some spatial structure: make neighboring pixels more similar in the channel dimension
    for i in range(1, H):
        example_tensor[:, i, :] = example_tensor[:, i, :] * 0.7 + example_tensor[:, i-1, :] * 0.3
    for j in range(1, W):
        example_tensor[:, :, j] = example_tensor[:, :, j] * 0.7 + example_tensor[:, :, j-1] * 0.3

    mode = 'shifted'
    """
    calculate mode:
    'shifted': used in paper `Transformer-based Transform Coding`, tend to be slower,
    'simple': more general spatial correlation to the center point
    """
    # 2. Set parameters
    center_y, center_x = H // 2, W // 2  # e.g., (16, 16)
    window_size = 5  # Look at a 5x5 neighborhood

    # 3. Calculate the correlation matrix
    if mode == 'simple':
        correlation_matrix = calculate_spatial_correlation_from_single_tensor(
        example_tensor, center_y, center_x, window_size
    )
    elif mode == 'shifted':
        correlation_matrix = calculate_average_spatial_autocorrelation_map(example_tensor, window_size)
    else:
        raise NotImplementedError

    print(f"Calculated {window_size}x{window_size} correlation map with center at ({center_y}, {center_x}):")
    print(correlation_matrix)

    plot_correlation_heatmap(correlation_matrix,
                            title=f"Spatial Correlation",
                            vmin_val=0.0, # Generally better to start from 0 when correlation is non-negative
                            )

