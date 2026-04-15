import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 
from PIL import Image 
import io
import os
import json

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
    point_size = np.random.choice([10, 20, 30, 50])
    
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
    density_type = np.random.choice(['uniform', 'clustered', 'mixed', 'curves'])
    
    # ===== GENERATE POINT COORDINATES =====
    
    data_coords = generate_point_coordinates(
        num_points=num_points,
        axis_range=axis_range,
        density_type=density_type,
        overlap_probability=overlap_probability,
        min_distance=min_distance
    )
    
    # ===== CHOOSE COLORS =====
    total_num_points = len(data_coords)
    colors = get_point_colors(total_num_points, color_scheme)
    
    # ===== CREATE PLOT =====
    
    fig, ax = plt.subplots(figsize=(image_size/100, image_size/100), dpi=100)
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)
    
    # Extract x and y coordinates
    x_coords = [coord[0] for coord in data_coords]
    y_coords = [coord[1] for coord in data_coords]
    
    # Plot scatter points
    # Legend (35% chance)
    show_legend = np.random.random() < 0.35
    
    if show_legend:
        num_classes  = np.random.randint(2, 4)
        class_names  = [f"Class {chr(65+i)}" for i in range(num_classes)]
        class_colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6'][:num_classes]
        point_classes = np.random.randint(0, num_classes, size=total_num_points)
        for cls_idx in range(num_classes):
            mask = point_classes == cls_idx
            ax.scatter([x for x, m in zip(x_coords, mask) if m],
                       [y for y, m in zip(y_coords, mask) if m],
                       s=point_size, c=class_colors[cls_idx],
                       alpha=0.8, edgecolors='none', label=class_names[cls_idx])
        ax.legend(loc=np.random.choice(['upper right', 'upper left', 'lower right', 'lower left']),
                  fontsize=6, markerscale=0.8, framealpha=0.7)
    else:
        ax.scatter(x_coords, y_coords, s=point_size, c=colors, alpha=0.8, edgecolors='none')

    if np.random.random() < 0.30:
        sorted_pairs = sorted(zip(x_coords, y_coords), key=lambda p: p[0])
        lx, ly = zip(*sorted_pairs)
        ls = np.random.choice(['-', '--', ':'])
        ax.plot(lx, ly, linestyle=ls, linewidth=np.random.uniform(0.5, 2.0),
                color=np.random.choice(['gray', 'black', '#aaaaaa']),
                alpha=np.random.uniform(0.3, 0.6), zorder=0)
    
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
    
    # ===== EXTRACT AXES BOUNDS BEFORE SAVING =====
    # We must draw the canvas first so bbox values are populated
    fig.canvas.draw()
    
    # Get the axes bounding box in figure pixels (at the figure's own dpi=100)
    renderer = fig.canvas.get_renderer()
    ax_bbox_display = ax.get_window_extent(renderer=renderer)  # in display (pixel) coords
    
    # Figure size in pixels at dpi=100
    fig_width_px  = fig.get_figwidth()  * fig.dpi   # == image_size
    fig_height_px = fig.get_figheight() * fig.dpi   # == image_size
    
    # Axes bounding box: x0, y0 (bottom-left) in display pixels, origin bottom-left
    ax_x0 = ax_bbox_display.x0   # left edge of plot area
    ax_y0 = ax_bbox_display.y0   # bottom edge of plot area  (matplotlib origin = bottom-left)
    ax_w  = ax_bbox_display.width
    ax_h  = ax_bbox_display.height

    # ===== CONVERT TO IMAGE =====
    
    # Save to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, facecolor=bg_color, bbox_inches='tight', pad_inches=0.05)
    buf.seek(0)
    img_before_resize = Image.open(buf).convert('RGB')
    plt.close(fig)
    
    # bbox_inches='tight' may change the actual saved image size, so we need to
    # record the saved size BEFORE resizing so we can scale the axes bounds accordingly.
    saved_w, saved_h = img_before_resize.size  # (width, height) in pixels
    
    # Scale factor from figure coords → saved image coords
    scale_x = saved_w / fig_width_px
    scale_y = saved_h / fig_height_px
    
    # Axes bounds in saved-image pixel space (origin still bottom-left here)
    saved_ax_x0 = ax_x0 * scale_x
    saved_ax_y0 = ax_y0 * scale_y
    saved_ax_w  = ax_w  * scale_x
    saved_ax_h  = ax_h  * scale_y
    
    # Resize to exact output size
    img = img_before_resize.resize((image_size, image_size), Image.LANCZOS)
    
    # Scale factor from saved-image → final output image
    out_scale_x = image_size / saved_w
    out_scale_y = image_size / saved_h
    
    # Final axes bounds in output image pixel space (origin: bottom-left)
    final_ax_x0 = saved_ax_x0 * out_scale_x
    final_ax_y0 = saved_ax_y0 * out_scale_y
    final_ax_w  = saved_ax_w  * out_scale_x
    final_ax_h  = saved_ax_h  * out_scale_y
    
    # ===== CONVERT DATA COORDS TO PIXEL COORDS =====
    # Now use the true axes bounds instead of the full image size
    
    pixel_coords = data_to_pixel_coords(
        data_coords=data_coords,
        axis_range=axis_range,
        image_size=image_size,
        ax_x0=final_ax_x0,
        ax_y0=final_ax_y0,
        ax_w=final_ax_w,
        ax_h=final_ax_h,
    )
    
    # Store parameters
    params = {
        'num_points': num_points,
        'point_size': point_size,
        'color_scheme': color_scheme,
        'show_grid': show_grid,
        'show_labels': show_labels,
        'bg_color': bg_color,
        'show_legend': show_legend,
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
    
    elif density_type == 'curves':
        # Generate points along curves (semicircle, parabola, sine wave, etc.)
        curve_types = ['semicircle', 'parabola', 'sine', 'exponential', 'ellipse']
        curve_type = np.random.choice(curve_types)
        
        # Random noise level for more realistic scatter
        noise_level = np.random.uniform(1, 4)
        
        if curve_type == 'semicircle':
            # Generate semicircle (top or bottom half)
            center_x = np.random.uniform(x_min + 20, x_max - 20)
            center_y = np.random.uniform(y_min + 20, y_max - 20)
            radius = np.random.uniform(15, 35)
            
            # Top or bottom semicircle
            if np.random.random() < 0.5:
                # Top semicircle
                angles = np.random.uniform(0, np.pi, num_points)
            else:
                # Bottom semicircle
                angles = np.random.uniform(np.pi, 2*np.pi, num_points)
            
            for angle in angles:
                x = center_x + radius * np.cos(angle) + np.random.normal(0, noise_level)
                y = center_y + radius * np.sin(angle) + np.random.normal(0, noise_level)
                x = np.clip(x, x_min, x_max)
                y = np.clip(y, y_min, y_max)
                data_coords.append([x, y])
        
        elif curve_type == 'parabola':
            # Generate parabola (upward or downward)
            x_vals = np.random.uniform(x_min + 10, x_max - 10, num_points)
            x_vals.sort()  # Sort for smoother appearance
            
            # Random parabola parameters
            a = np.random.uniform(-0.05, 0.05)
            if abs(a) < 0.01:
                a = 0.02 if np.random.random() < 0.5 else -0.02
            
            vertex_x = np.random.uniform(x_min + 20, x_max - 20)
            vertex_y = np.random.uniform(y_min + 20, y_max - 20)
            
            for x in x_vals:
                y = a * (x - vertex_x)**2 + vertex_y + np.random.normal(0, noise_level)
                y = np.clip(y, y_min, y_max)
                data_coords.append([x, y])
        
        elif curve_type == 'sine':
            # Generate sine wave
            x_vals = np.linspace(x_min + 5, x_max - 5, num_points)
            
            # Random sine parameters
            amplitude = np.random.uniform(10, 25)
            frequency = np.random.uniform(0.05, 0.2)
            phase = np.random.uniform(0, 2*np.pi)
            vertical_shift = (y_min + y_max) / 2
            
            for x in x_vals:
                y = amplitude * np.sin(frequency * x + phase) + vertical_shift
                y += np.random.normal(0, noise_level)
                y = np.clip(y, y_min, y_max)
                data_coords.append([x, y])
        
        elif curve_type == 'exponential':
            # Generate exponential curve (growth or decay)
            x_vals = np.random.uniform(x_min + 10, x_max - 10, num_points)
            x_vals.sort()
            
            # Random exponential parameters
            if np.random.random() < 0.5:
                # Growth
                base = np.random.uniform(1.02, 1.08)
                y_offset = y_min + 10
            else:
                # Decay
                base = np.random.uniform(0.92, 0.98)
                y_offset = y_max - 10
            
            scale = np.random.uniform(0.5, 2)
            
            for x in x_vals:
                y = scale * (base ** x) + y_offset + np.random.normal(0, noise_level)
                y = np.clip(y, y_min, y_max)
                data_coords.append([x, y])
        
        elif curve_type == 'ellipse':
            # Generate ellipse (full or partial)
            center_x = np.random.uniform(x_min + 25, x_max - 25)
            center_y = np.random.uniform(y_min + 25, y_max - 25)
            a_axis = np.random.uniform(15, 30)  # Semi-major axis
            b_axis = np.random.uniform(10, 25)  # Semi-minor axis
            
            # Full ellipse or partial arc
            if np.random.random() < 0.5:
                # Full ellipse
                angles = np.random.uniform(0, 2*np.pi, num_points)
            else:
                # Partial arc
                start_angle = np.random.uniform(0, np.pi)
                end_angle = start_angle + np.random.uniform(np.pi/2, 3*np.pi/2)
                angles = np.random.uniform(start_angle, end_angle, num_points)
            
            for angle in angles:
                x = center_x + a_axis * np.cos(angle) + np.random.normal(0, noise_level)
                y = center_y + b_axis * np.sin(angle) + np.random.normal(0, noise_level)
                x = np.clip(x, x_min, x_max)
                y = np.clip(y, y_min, y_max)
                data_coords.append([x, y])
    
    elif density_type == 'mixed':
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
    
    else:
        # Default to uniform if unknown type
        for _ in range(num_points):
            x = np.random.uniform(x_min, x_max)
            y = np.random.uniform(y_min, y_max)
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


def data_to_pixel_coords(data_coords, axis_range, image_size,
                          ax_x0=None, ax_y0=None, ax_w=None, ax_h=None):
    """
    Convert data coordinates to pixel coordinates in the output image.

    When ax_* parameters are provided (recommended), they define the exact
    bounding box of the matplotlib axes in the final output image so that
    pixel coords align with where dots were actually drawn.

    ax_x0, ax_y0  – left/bottom edge of the axes in output-image pixels,
                    with origin at the BOTTOM-LEFT (matplotlib convention).
    ax_w, ax_h    – width/height of the axes area in output-image pixels.

    If ax_* are not provided, falls back to the old behaviour (full image).
    """
    x_min, x_max = axis_range
    y_min, y_max = axis_range

    # Fall back to full-image mapping when no axes bounds are given
    if ax_x0 is None:
        ax_x0, ax_y0, ax_w, ax_h = 0, 0, image_size, image_size

    pixel_coords = []
    for x_data, y_data in data_coords:
        # Normalise data value to [0, 1]
        x_norm = (x_data - x_min) / (x_max - x_min)
        y_norm = (y_data - y_min) / (y_max - y_min)

        # Map into the axes pixel region
        x_pixel = ax_x0 + x_norm * ax_w

        # y in matplotlib increases upward; in image pixels it increases downward.
        # ax_y0 is the BOTTOM of the axes in bottom-left-origin space.
        # Convert to top-left-origin: image_size - (ax_y0 + y_norm * ax_h)
        y_pixel = image_size - (ax_y0 + y_norm * ax_h)

        pixel_coords.append([x_pixel, y_pixel])

    return pixel_coords

def convert_numpy(obj):
    if isinstance(obj, np.generic):  
        return obj.item()            
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, list):
        return [convert_numpy(x) for x in obj]
    elif isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    else:
        return obj


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
        json.dump(convert_numpy(annotations), f, indent=2)
    
    print(f"Generated {num_samples} {split} samples")
    print(f"Saved to {output_dir}")
    print(f"Metadata saved to {metadata_path}")

def save_image(img, path):
    """Save image (PIL Image or numpy array) to path."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    if isinstance(img, Image.Image):
        # PIL Image
        img.save(path)
    elif isinstance(img, np.ndarray):
        # Numpy array (heatmap)
        # Convert to 0-255 range and save as grayscale
        img_uint8 = (img * 255).astype(np.uint8)
        Image.fromarray(img_uint8, mode='L').save(path)
    else:
        raise ValueError(f"Unknown image type: {type(img)}")

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
        num_samples=config.NUM_TEST,
        output_dir="/scratch/gssodhi/data_extract/test",
        split="test"
    )

# Generate the dataset
main()