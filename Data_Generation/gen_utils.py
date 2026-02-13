import numpy as np
import matplotlib.pyplot as plt 
matplotlib.use('Agg')
from PIL import Image 
import io

import gen_config as config

def generate_heatmap(image_size, point_coords_pixel, sigma):
    H, W = image_size
    heatmap = np.zeros((H, W), dtype=np.float32)
    
    # Define Gaussian kernel size (3-sigma rule)
    kernel_radius = int(3 * sigma)
    
    for (x, y) in point_coords_pixel:
        # Skip if point is outside image
        if x < 0 or x >= W or y < 0 or y >= H:
            continue
        
        # Define region of interest
        x_min = max(0, int(x) - kernel_radius)
        x_max = min(W, int(x) + kernel_radius + 1)
        y_min = max(0, int(y) - kernel_radius)
        y_max = min(H, int(y) + kernel_radius + 1)
        
        # Create local coordinate grids
        y_local, x_local = np.ogrid[y_min:y_max, x_min:x_max]
        
        # Calculate Gaussian only in local region
        gaussian = np.exp(-((x_local - x)**2 + (y_local - y)**2) / (2 * sigma**2))
        
        # Update heatmap (max to preserve peaks)
        heatmap[y_min:y_max, x_min:x_max] = np.maximum(
            heatmap[y_min:y_max, x_min:x_max], 
            gaussian
        )
    
    return heatmap

def generate_scatter_plot(
    image_size=224,
    min_points=50,
    max_points=300,
    axis_range=(0, 100),
    min_distance=3,  # Minimum distance in pixels for overlap control
    overlap_probability=0.3
):
    """
    Generate a synthetic scatter plot with random variations.
    
    Returns:
        img: PIL Image of the scatter plot
        data_coords: List of (x, y) in data coordinates
        pixel_coords: List of (x, y) in pixel coordinates
        params: Dictionary of generation parameters
    """
    
    # ===== RANDOM CHOICES =====
    
    # 1. Number of points
    num_points = np.random.randint(min_points, max_points + 1)
    
    # 2. Point size (s parameter in scatter, which is area in points^2)
    point_size = np.random.choice([10, 20, 30, 50, 80, 100])
    
    # 3. Color scheme
    color_schemes = [
        'single_red', 'single_blue', 'single_green', 'single_black',
        'single_purple', 'single_orange', 'single_cyan',
        'multi_color', 'gradient'
    ]
    color_scheme = np.random.choice(color_schemes)
    
    # 4. Grid
    show_grid = np.random.choice([True, False])
    
    # 5. Axis labels
    show_labels = np.random.choice([True, False])
    
    # 6. Background color
    background_colors = ['white', 'lightgray', '#f0f0f0', '#fafafa']
    bg_color = np.random.choice(background_colors)
    
    # 7. Density variation (clustered vs uniform)
    density_type = np.random.choice(['uniform', 'clustered', 'mixed'])
    
    # ===== GENERATE POINT COORDINATES =====
    
    data_coords = generate_point_coordinates(
        num_points=num_points,
        axis_range=axis_range,
        density_type=density_type,
        overlap_probability=overlap_probability,
        min_distance=min_distance
    )
    
    # ===== CHOOSE COLORS =====
    
    colors = get_point_colors(num_points, color_scheme)
    
    # ===== CREATE PLOT =====
    
    fig, ax = plt.subplots(figsize=(image_size/100, image_size/100), dpi=100)
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)
    
    # Extract x and y coordinates
    x_coords = [coord[0] for coord in data_coords]
    y_coords = [coord[1] for coord in data_coords]
    
    # Plot scatter points
    ax.scatter(x_coords, y_coords, s=point_size, c=colors, alpha=0.8, edgecolors='none')
    
    # Set axis limits
    ax.set_xlim(axis_range[0], axis_range[1])
    ax.set_ylim(axis_range[0], axis_range[1])
    
    # Grid
    if show_grid:
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    else:
        ax.grid(False)
    
    # Labels
    if show_labels:
        ax.set_xlabel('X', fontsize=8)
        ax.set_ylabel('Y', fontsize=8)
        ax.tick_params(labelsize=6)
    else:
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Remove spines for cleaner look
    if np.random.random() < 0.3:  # 30% chance
        for spine in ax.spines.values():
            spine.set_visible(False)
    
    plt.tight_layout(pad=0.1)
    
    # ===== CONVERT TO IMAGE =====
    
    # Save to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, facecolor=bg_color, bbox_inches='tight', pad_inches=0.05)
    buf.seek(0)
    img = Image.open(buf).convert('RGB')
    plt.close(fig)
    
    # Resize to exact size
    img = img.resize((image_size, image_size), Image.LANCZOS)
    
    # ===== CONVERT DATA COORDS TO PIXEL COORDS =====
    
    pixel_coords = data_to_pixel_coords(
        data_coords=data_coords,
        axis_range=axis_range,
        image_size=image_size
    )
    
    # Store parameters
    params = {
        'num_points': num_points,
        'point_size': point_size,
        'color_scheme': color_scheme,
        'show_grid': show_grid,
        'show_labels': show_labels,
        'bg_color': bg_color,
        'density_type': density_type
    }
    
    return img, data_coords, pixel_coords, params


def generate_point_coordinates(num_points, axis_range, density_type, overlap_probability, min_distance):
    """
    Generate point coordinates with different density patterns.
    """
    data_coords = []
    x_min, x_max = axis_range
    y_min, y_max = axis_range
    
    if density_type == 'uniform':
        # Completely random uniform distribution
        for _ in range(num_points):
            x = np.random.uniform(x_min, x_max)
            y = np.random.uniform(y_min, y_max)
            data_coords.append([x, y])
    
    elif density_type == 'clustered':
        # Create 2-4 clusters
        num_clusters = np.random.randint(2, 5)
        points_per_cluster = num_points // num_clusters
        
        # Generate cluster centers
        cluster_centers = []
        for _ in range(num_clusters):
            cx = np.random.uniform(x_min + 10, x_max - 10)
            cy = np.random.uniform(y_min + 10, y_max - 10)
            cluster_centers.append([cx, cy])
        
        # Generate points around clusters
        for i, (cx, cy) in enumerate(cluster_centers):
            n = points_per_cluster if i < num_clusters - 1 else num_points - len(data_coords)
            cluster_std = np.random.uniform(3, 8)
            
            for _ in range(n):
                x = np.random.normal(cx, cluster_std)
                y = np.random.normal(cy, cluster_std)
                # Clamp to axis range
                x = np.clip(x, x_min, x_max)
                y = np.clip(y, y_min, y_max)
                data_coords.append([x, y])
    
    else:  # mixed
        # Half uniform, half clustered
        half = num_points // 2
        
        # Uniform half
        for _ in range(half):
            x = np.random.uniform(x_min, x_max)
            y = np.random.uniform(y_min, y_max)
            data_coords.append([x, y])
        
        # Clustered half
        num_clusters = np.random.randint(1, 3)
        points_per_cluster = (num_points - half) // num_clusters
        
        for i in range(num_clusters):
            cx = np.random.uniform(x_min + 10, x_max - 10)
            cy = np.random.uniform(y_min + 10, y_max - 10)
            cluster_std = np.random.uniform(3, 8)
            
            n = points_per_cluster if i < num_clusters - 1 else num_points - len(data_coords)
            for _ in range(n):
                x = np.random.normal(cx, cluster_std)
                y = np.random.normal(cy, cluster_std)
                x = np.clip(x, x_min, x_max)
                y = np.clip(y, y_min, y_max)
                data_coords.append([x, y])
    
    # create intentional overlaps
    if np.random.random() < overlap_probability:
        num_overlaps = np.random.randint(5, 20)
        for _ in range(num_overlaps):
            # Pick a random existing point
            base_point = data_coords[np.random.randint(0, len(data_coords))]
            # Add a nearby point (controlled overlap)
            offset_x = np.random.uniform(-2, 2)
            offset_y = np.random.uniform(-2, 2)
            new_x = np.clip(base_point[0] + offset_x, x_min, x_max)
            new_y = np.clip(base_point[1] + offset_y, y_min, y_max)
            data_coords.append([new_x, new_y])
    
    return data_coords


def get_point_colors(num_points, color_scheme):
    """
    Generate colors based on chosen scheme.
    """
    if color_scheme.startswith('single_'):
        color_name = color_scheme.split('_')[1]
        color_map = {
            'red': '#e74c3c',
            'blue': '#3498db',
            'green': '#2ecc71',
            'black': '#2c3e50',
            'purple': '#9b59b6',
            'orange': '#e67e22',
            'cyan': '#1abc9c'
        }
        return [color_map[color_name]] * num_points
    
    elif color_scheme == 'multi_color':
        # Random colors for each point
        colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#e67e22', '#1abc9c']
        return [np.random.choice(colors) for _ in range(num_points)]
    
    elif color_scheme == 'gradient':
        # Color gradient based on position
        cmap = plt.cm.get_cmap(np.random.choice(['viridis', 'plasma', 'coolwarm', 'RdYlBu']))
        return [cmap(i / num_points) for i in range(num_points)]
    
    return ['blue'] * num_points


def data_to_pixel_coords(data_coords, axis_range, image_size):
    """
    Convert data coordinates to pixel coordinates.
    
    Args:
        data_coords: List of [x, y] in data space
        axis_range: (min, max) of the axes
        image_size: Size of output image
    
    Returns:
        pixel_coords: List of [x_pixel, y_pixel]
    """
    x_min, x_max = axis_range
    y_min, y_max = axis_range
    
    pixel_coords = []
    for x_data, y_data in data_coords:
        # Normalize to [0, 1]
        x_norm = (x_data - x_min) / (x_max - x_min)
        y_norm = (y_data - y_min) / (y_max - y_min)
        
        # Convert to pixels
        # Note: y-axis is flipped in images (origin at top-left)
        x_pixel = x_norm * image_size
        y_pixel = (1 - y_norm) * image_size  # Flip y
        
        pixel_coords.append([x_pixel, y_pixel])
    
    return pixel_coords

def generate_dataset(num_samples, output_dir, split):
    annotations = {}
    
    for i in range(num_samples):
        # Generate scatter plot and get point coordinates
        img, data_coords, pixel_coords, params = generate_scatter_plot(
            image_size=config.IMAGE_SIZE,
            min_points=config.MIN_POINTS,
            max_points=config.MAX_POINTS,
            axis_range=config.AXIS_RANGE,
            min_distance=config.MIN_DISTANCE,
            overlap_probability=config.OVERLAP_PROBABILITY
        )
        
        # Generate corresponding heatmap
        H, W = config.IMAGE_SIZE, config.IMAGE_SIZE
        heatmap = generate_heatmap((H, W), pixel_coords, sigma=config.GAUSSIAN_SIGMA)
        
        # Save image and heatmap
        img_filename = f"{split}_{i:05d}.png"
        img_path = f"{output_dir}/images/{img_filename}"
        heatmap_path = f"{output_dir}/heatmaps/{img_filename}"
        
        save_image(img, img_path)
        save_image(heatmap, heatmap_path)
        
        # Store metadata
        annotations[img_filename] = {
            "num_points": len(pixel_coords),
            "pixel_coords": pixel_coords,
            "image_size": [H, W],
            "params": params,
            "data_coords": data_coords,
            "gaussian_sigma": config.GAUSSIAN_SIGMA,
            "axis_range": config.AXIS_RANGE,
        }
    
    # Save annotations
    metadata_path = f"/scratch/gssodhi/data_extract/metadata/{split}_annotations.json"
    os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
    with open(metadata_path, 'w') as f:
        json.dump(annotations, f, indent=2)
    
    print(f"Generated {num_samples} {split} samples")
    print(f"Saved to {output_dir}")
    print(f"Metadata saved to {metadata_path}")

# Generate the dataset
def main():
    # Set random seed for reproducibility
    np.random.seed(42)  

    # Generate training data
    print("Generating training data...")
    generate_dataset(
        num_samples=config.NUM_TRAIN,
        output_dir="/scratch/gssodhi/data_extract/train",
        split="train"
    )
    
    # Generate validation data
    print("Generating validation data...")
    generate_dataset(
        num_samples=config.NUM_VAL,
        output_dir="/scratch/gssodhi/data_extract/val",
        split="val"
    )
    
    # Generate test data
    print("Generating test data...")
    generate_dataset(
        num_samples=configNUM_TEST,
        output_dir="/scratch/gssodhi/data_extract/test",
        split="test"
    )

# Generate the dataset
main()