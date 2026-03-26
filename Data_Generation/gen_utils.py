import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 
from PIL import Image 
import io
import os
import json

from tqdm import tqdm

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
    min_distance=3,
    overlap_probability=0.3
):
    """
    Generate a synthetic scatter plot with random variations.
    Supports: plain scatter, scatter+line, legends, error bars,
              log-scale axes, annotations, varying marker shapes,
              low-contrast points, and bubble (size-encoded) charts.

    Returns:
        img: PIL Image of the scatter plot
        data_coords: List of (x, y) in data coordinates
        pixel_coords: List of (x, y) in pixel coordinates
        params: Dictionary of generation parameters
    """

    # ===== RANDOM CHOICES =====

    num_points   = np.random.randint(min_points, max_points + 1)
    point_size   = np.random.choice([10, 20, 30, 50, 80, 100])
    color_schemes = [
        'single_red', 'single_blue', 'single_green', 'single_black',
        'single_purple', 'single_orange', 'single_cyan',
        'multi_color', 'gradient'
    ]
    color_scheme  = np.random.choice(color_schemes)
    show_grid     = np.random.choice([True, False])
    show_labels   = np.random.choice([True, False])
    background_colors = ['white', 'lightgray', '#f0f0f0', '#fafafa']
    bg_color      = np.random.choice(background_colors)
    density_type  = np.random.choice(['uniform', 'clustered', 'mixed', 'curves'])

    # ---------- NEW edge-case flags ----------
    # Lines connecting points (30 % chance)
    show_lines     = np.random.random() < 0.30

    # Legend (35 % chance; only meaningful when we have class colours)
    show_legend    = np.random.random() < 0.35

    # Error bars (20 % chance)
    show_error_bars = np.random.random() < 0.20

    # Log-scale axes (15 % chance)
    use_log_scale   = np.random.random() < 0.15

    # Annotations on a random subset of points (15 % chance)
    show_annotations = np.random.random() < 0.15

    # Bubble chart – point size encodes a third variable (20 % chance)
    use_bubble_size  = np.random.random() < 0.20

    # Varying marker shapes (25 % chance)
    use_varied_markers = np.random.random() < 0.25

    # Low-contrast colour (10 % chance – stress test)
    use_low_contrast   = np.random.random() < 0.10
    # -----------------------------------------

    # ===== GENERATE POINT COORDINATES =====
    data_coords = generate_point_coordinates(
        num_points=num_points,
        axis_range=axis_range,
        density_type=density_type,
        overlap_probability=overlap_probability,
        min_distance=min_distance
    )
    total_num_points = len(data_coords)

    x_coords = [c[0] for c in data_coords]
    y_coords = [c[1] for c in data_coords]

    # ===== LEGEND / CLASS SETUP =====
    # When showing a legend we assign each point a class with its own colour.
    # This replaces the flat colour list for that subset of runs.
    if show_legend:
        num_classes   = np.random.randint(2, 5)
        class_names   = [f"Class {chr(65+i)}" for i in range(num_classes)]
        class_palette = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#e67e22']
        class_colors  = class_palette[:num_classes]
        point_classes = np.random.randint(0, num_classes, size=total_num_points)
        # per-point colour list (used for error bars / annotations lookup)
        per_point_colors = [class_colors[c] for c in point_classes]
    else:
        per_point_colors = get_point_colors(total_num_points, color_scheme)
        point_classes    = None
        class_names      = None
        class_colors     = None

    # ===== BUBBLE SIZES =====
    if use_bubble_size:
        # Each point gets a random size drawn from a wide range
        bubble_sizes = np.random.uniform(10, 300, size=total_num_points).tolist()
    else:
        bubble_sizes = [point_size] * total_num_points

    # ===== MARKER SHAPES =====
    all_markers = ['o', 's', '^', 'D', 'v', 'P', '*', 'X']
    if use_varied_markers:
        point_markers = [np.random.choice(all_markers) for _ in range(total_num_points)]
    else:
        point_markers = ['o'] * total_num_points

    # ===== LOW-CONTRAST COLOURS =====
    if use_low_contrast:
        low_contrast_map = {
            'white':   ['#e0e0e0', '#d5d5d5', '#c8c8c8'],
            'lightgray': ['#b0b0b0', '#a8a8a8', '#bcbcbc'],
            '#f0f0f0': ['#dcdcdc', '#d0d0d0', '#e8e8e8'],
            '#fafafa': ['#e8e8e8', '#ededef', '#dfe0e1'],
        }
        lc_color = np.random.choice(low_contrast_map.get(bg_color, ['#cccccc']))
        per_point_colors = [lc_color] * total_num_points

    # ===== CREATE PLOT =====
    fig, ax = plt.subplots(figsize=(image_size/100, image_size/100), dpi=100)
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)

    # --- 1. LINES (drawn first so they sit behind the scatter points) ---
    if show_lines:
        line_style  = np.random.choice(['solid', 'dashed', 'dotted'])
        line_width  = np.random.uniform(0.5, 2.0)
        line_color  = np.random.choice(['gray', 'black', '#888888', '#aaaaaa'])
        line_alpha  = np.random.uniform(0.3, 0.6)
        sort_by_x   = np.random.random() < 0.7          # usually sort by x for tidiness

        if sort_by_x:
            order = np.argsort(x_coords)
            lx = [x_coords[i] for i in order]
            ly = [y_coords[i] for i in order]
        else:
            lx, ly = x_coords, y_coords

        ls_map = {'solid': '-', 'dashed': '--', 'dotted': ':'}
        ax.plot(lx, ly,
                linestyle=ls_map[line_style],
                linewidth=line_width,
                color=line_color,
                alpha=line_alpha,
                zorder=0)          # behind scatter dots

    # --- 2. SCATTER POINTS ---
    alpha = np.random.uniform(0.4, 0.95)   # randomise alpha (was fixed 0.8)

    if show_legend and class_names is not None:
        # Draw one scatter call per class so matplotlib auto-generates legend handles
        for cls_idx in range(len(class_names)):
            mask = point_classes == cls_idx
            cx = [x for x, m in zip(x_coords, mask) if m]
            cy = [y for y, m in zip(y_coords, mask) if m]
            cs = [s for s, m in zip(bubble_sizes, mask) if m]
            if not cx:
                continue
            # Use a single marker per class (pick first in list for simplicity)
            ax.scatter(cx, cy,
                       s=cs,
                       c=class_colors[cls_idx],
                       alpha=alpha,
                       edgecolors='none',
                       label=class_names[cls_idx],
                       zorder=2)
    elif use_varied_markers:
        # Draw point-by-point when markers vary (matplotlib scatter doesn't accept a list of markers)
        for xi, yi, si, ci, mi in zip(x_coords, y_coords, bubble_sizes, per_point_colors, point_markers):
            ax.scatter(xi, yi, s=si, c=ci, alpha=alpha, edgecolors='none', marker=mi, zorder=2)
    else:
        ax.scatter(x_coords, y_coords,
                   s=bubble_sizes,
                   c=per_point_colors,
                   alpha=alpha,
                   edgecolors='none',
                   zorder=2)

    # --- 3. ERROR BARS ---
    if show_error_bars:
        xerr_vals = np.random.uniform(0.5, 4.0, size=total_num_points)
        yerr_vals = np.random.uniform(0.5, 4.0, size=total_num_points)
        eb_color  = np.random.choice(['gray', 'black', '#555555'])
        ax.errorbar(x_coords, y_coords,
                    xerr=xerr_vals, yerr=yerr_vals,
                    fmt='none',
                    ecolor=eb_color,
                    elinewidth=0.6,
                    capsize=1.5,
                    alpha=0.5,
                    zorder=1)      # between lines and scatter dots

    # --- 4. ANNOTATIONS ---
    if show_annotations:
        n_annotate = np.random.randint(3, min(12, total_num_points + 1))
        chosen_idx = np.random.choice(total_num_points, size=n_annotate, replace=False)
        for idx in chosen_idx:
            label_text = np.random.choice([
                f"P{idx}", f"({x_coords[idx]:.0f},{y_coords[idx]:.0f})",
                f"n={idx}", chr(65 + (idx % 26))
            ])
            ax.annotate(label_text,
                        (x_coords[idx], y_coords[idx]),
                        fontsize=np.random.randint(4, 7),
                        xytext=(np.random.uniform(2, 6), np.random.uniform(2, 6)),
                        textcoords='offset points',
                        color='#333333',
                        alpha=0.75)

    # --- 5. AXIS LIMITS & SCALE ---
    x_min, x_max = axis_range
    y_min, y_max = axis_range

    if use_log_scale:
        # Log scale requires positive values; shift range to [1, 101]
        ax.set_xlim(1, x_max + 1)
        ax.set_ylim(1, y_max + 1)
        ax.set_xscale('log')
        ax.set_yscale('log')
    else:
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

    # --- 6. GRID ---
    if show_grid:
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    else:
        ax.grid(False)

    # --- 7. LABELS / TICKS ---
    if show_labels:
        ax.set_xlabel('X', fontsize=8)
        ax.set_ylabel('Y', fontsize=8)
        ax.tick_params(labelsize=6)
    else:
        ax.set_xticks([])
        ax.set_yticks([])

    # --- 8. SPINES ---
    if np.random.random() < 0.3:
        for spine in ax.spines.values():
            spine.set_visible(False)

    # --- 9. LEGEND ---
    if show_legend and class_names is not None:
        legend_loc = np.random.choice(['upper right', 'upper left', 'lower right', 'lower left'])
        ax.legend(loc=legend_loc, fontsize=6, markerscale=0.8,
                  framealpha=np.random.uniform(0.5, 0.9))

    plt.tight_layout(pad=0.1)

    # ===== EXTRACT AXES BOUNDS BEFORE SAVING =====
    fig.canvas.draw()
    renderer        = fig.canvas.get_renderer()
    ax_bbox_display = ax.get_window_extent(renderer=renderer)

    fig_width_px    = fig.get_figwidth()  * fig.dpi
    fig_height_px   = fig.get_figheight() * fig.dpi

    ax_x0 = ax_bbox_display.x0
    ax_y0 = ax_bbox_display.y0
    ax_w  = ax_bbox_display.width
    ax_h  = ax_bbox_display.height

    # ===== CONVERT TO IMAGE =====
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, facecolor=bg_color,
                bbox_inches='tight', pad_inches=0.05)
    buf.seek(0)
    img_before_resize = Image.open(buf).convert('RGB')
    plt.close(fig)

    saved_w, saved_h = img_before_resize.size

    scale_x = saved_w / fig_width_px
    scale_y = saved_h / fig_height_px

    saved_ax_x0 = ax_x0 * scale_x
    saved_ax_y0 = ax_y0 * scale_y
    saved_ax_w  = ax_w  * scale_x
    saved_ax_h  = ax_h  * scale_y

    img = img_before_resize.resize((image_size, image_size), Image.LANCZOS)

    out_scale_x = image_size / saved_w
    out_scale_y = image_size / saved_h

    final_ax_x0 = saved_ax_x0 * out_scale_x
    final_ax_y0 = saved_ax_y0 * out_scale_y
    final_ax_w  = saved_ax_w  * out_scale_x
    final_ax_h  = saved_ax_h  * out_scale_y

    # ===== CONVERT DATA COORDS TO PIXEL COORDS =====
    pixel_coords = data_to_pixel_coords(
        data_coords=data_coords,
        axis_range=axis_range,
        image_size=image_size,
        ax_x0=final_ax_x0,
        ax_y0=final_ax_y0,
        ax_w=final_ax_w,
        ax_h=final_ax_h,
    )

    # ===== STORE PARAMETERS =====
    params = {
        'num_points':        num_points,
        'point_size':        point_size,
        'color_scheme':      color_scheme,
        'show_grid':         show_grid,
        'show_labels':       show_labels,
        'bg_color':          bg_color,
        'density_type':      density_type,
        # new flags
        'show_lines':        show_lines,
        'show_legend':       show_legend,
        'show_error_bars':   show_error_bars,
        'use_log_scale':     use_log_scale,
        'show_annotations':  show_annotations,
        'use_bubble_size':   use_bubble_size,
        'use_varied_markers':use_varied_markers,
        'use_low_contrast':  use_low_contrast,
    }

    return img, data_coords, pixel_coords, params


# ============================================================
#  All helper functions below are unchanged from the original
# ============================================================

def generate_point_coordinates(num_points, axis_range, density_type, overlap_probability, min_distance):
    data_coords = []
    x_min, x_max = axis_range
    y_min, y_max = axis_range
    
    if density_type == 'uniform':
        for _ in range(num_points):
            x = np.random.uniform(x_min, x_max)
            y = np.random.uniform(y_min, y_max)
            data_coords.append([x, y])
    
    elif density_type == 'clustered':
        num_clusters = np.random.randint(2, 5)
        points_per_cluster = num_points // num_clusters
        cluster_centers = []
        for _ in range(num_clusters):
            cx = np.random.uniform(x_min + 10, x_max - 10)
            cy = np.random.uniform(y_min + 10, y_max - 10)
            cluster_centers.append([cx, cy])
        for i, (cx, cy) in enumerate(cluster_centers):
            n = points_per_cluster if i < num_clusters - 1 else num_points - len(data_coords)
            cluster_std = np.random.uniform(3, 8)
            for _ in range(n):
                x = np.clip(np.random.normal(cx, cluster_std), x_min, x_max)
                y = np.clip(np.random.normal(cy, cluster_std), y_min, y_max)
                data_coords.append([x, y])
    
    elif density_type == 'curves':
        curve_types = ['semicircle', 'parabola', 'sine', 'exponential', 'ellipse']
        curve_type  = np.random.choice(curve_types)
        noise_level = np.random.uniform(1, 4)
        
        if curve_type == 'semicircle':
            center_x = np.random.uniform(x_min + 20, x_max - 20)
            center_y = np.random.uniform(y_min + 20, y_max - 20)
            radius   = np.random.uniform(15, 35)
            angles   = (np.random.uniform(0, np.pi, num_points)
                        if np.random.random() < 0.5
                        else np.random.uniform(np.pi, 2*np.pi, num_points))
            for angle in angles:
                x = np.clip(center_x + radius*np.cos(angle) + np.random.normal(0, noise_level), x_min, x_max)
                y = np.clip(center_y + radius*np.sin(angle) + np.random.normal(0, noise_level), y_min, y_max)
                data_coords.append([x, y])
        
        elif curve_type == 'parabola':
            x_vals   = np.sort(np.random.uniform(x_min+10, x_max-10, num_points))
            a        = np.random.uniform(-0.05, 0.05)
            if abs(a) < 0.01: a = 0.02 if np.random.random() < 0.5 else -0.02
            vertex_x = np.random.uniform(x_min+20, x_max-20)
            vertex_y = np.random.uniform(y_min+20, y_max-20)
            for x in x_vals:
                y = np.clip(a*(x-vertex_x)**2 + vertex_y + np.random.normal(0, noise_level), y_min, y_max)
                data_coords.append([x, y])
        
        elif curve_type == 'sine':
            x_vals        = np.linspace(x_min+5, x_max-5, num_points)
            amplitude     = np.random.uniform(10, 25)
            frequency     = np.random.uniform(0.05, 0.2)
            phase         = np.random.uniform(0, 2*np.pi)
            vertical_shift = (y_min + y_max) / 2
            for x in x_vals:
                y = np.clip(amplitude*np.sin(frequency*x+phase) + vertical_shift + np.random.normal(0, noise_level), y_min, y_max)
                data_coords.append([x, y])
        
        elif curve_type == 'exponential':
            x_vals = np.sort(np.random.uniform(x_min+10, x_max-10, num_points))
            if np.random.random() < 0.5:
                base, y_offset = np.random.uniform(1.02, 1.08), y_min+10
            else:
                base, y_offset = np.random.uniform(0.92, 0.98), y_max-10
            scale = np.random.uniform(0.5, 2)
            for x in x_vals:
                y = np.clip(scale*(base**x) + y_offset + np.random.normal(0, noise_level), y_min, y_max)
                data_coords.append([x, y])
        
        elif curve_type == 'ellipse':
            center_x = np.random.uniform(x_min+25, x_max-25)
            center_y = np.random.uniform(y_min+25, y_max-25)
            a_axis   = np.random.uniform(15, 30)
            b_axis   = np.random.uniform(10, 25)
            if np.random.random() < 0.5:
                angles = np.random.uniform(0, 2*np.pi, num_points)
            else:
                start = np.random.uniform(0, np.pi)
                end   = start + np.random.uniform(np.pi/2, 3*np.pi/2)
                angles = np.random.uniform(start, end, num_points)
            for angle in angles:
                x = np.clip(center_x + a_axis*np.cos(angle) + np.random.normal(0, noise_level), x_min, x_max)
                y = np.clip(center_y + b_axis*np.sin(angle) + np.random.normal(0, noise_level), y_min, y_max)
                data_coords.append([x, y])
    
    elif density_type == 'mixed':
        half = num_points // 2
        for _ in range(half):
            data_coords.append([np.random.uniform(x_min, x_max), np.random.uniform(y_min, y_max)])
        num_clusters = np.random.randint(1, 3)
        ppc = (num_points - half) // num_clusters
        for i in range(num_clusters):
            cx  = np.random.uniform(x_min+10, x_max-10)
            cy  = np.random.uniform(y_min+10, y_max-10)
            std = np.random.uniform(3, 8)
            n   = ppc if i < num_clusters-1 else num_points - len(data_coords)
            for _ in range(n):
                x = np.clip(np.random.normal(cx, std), x_min, x_max)
                y = np.clip(np.random.normal(cy, std), y_min, y_max)
                data_coords.append([x, y])
    
    else:
        for _ in range(num_points):
            data_coords.append([np.random.uniform(x_min, x_max), np.random.uniform(y_min, y_max)])
    
    # Intentional overlaps
    if np.random.random() < overlap_probability:
        for _ in range(np.random.randint(5, 20)):
            base  = data_coords[np.random.randint(0, len(data_coords))]
            new_x = np.clip(base[0] + np.random.uniform(-2, 2), x_min, x_max)
            new_y = np.clip(base[1] + np.random.uniform(-2, 2), y_min, y_max)
            data_coords.append([new_x, new_y])
    
    return data_coords


def get_point_colors(num_points, color_scheme):
    if color_scheme.startswith('single_'):
        color_map = {
            'red': '#e74c3c', 'blue': '#3498db', 'green': '#2ecc71',
            'black': '#2c3e50', 'purple': '#9b59b6', 'orange': '#e67e22', 'cyan': '#1abc9c'
        }
        return [color_map[color_scheme.split('_')[1]]] * num_points
    elif color_scheme == 'multi_color':
        colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#e67e22', '#1abc9c']
        return [np.random.choice(colors) for _ in range(num_points)]
    elif color_scheme == 'gradient':
        cmap = plt.cm.get_cmap(np.random.choice(['viridis', 'plasma', 'coolwarm', 'RdYlBu']))
        return [cmap(i / num_points) for i in range(num_points)]
    return ['blue'] * num_points


def data_to_pixel_coords(data_coords, axis_range, image_size,
                          ax_x0=None, ax_y0=None, ax_w=None, ax_h=None):
    x_min, x_max = axis_range
    y_min, y_max = axis_range
    if ax_x0 is None:
        ax_x0, ax_y0, ax_w, ax_h = 0, 0, image_size, image_size
    pixel_coords = []
    for x_data, y_data in data_coords:
        x_norm   = (x_data - x_min) / (x_max - x_min)
        y_norm   = (y_data - y_min) / (y_max - y_min)
        x_pixel  = ax_x0 + x_norm * ax_w
        y_pixel  = image_size - (ax_y0 + y_norm * ax_h)
        pixel_coords.append([x_pixel, y_pixel])
    return pixel_coords


def convert_numpy(obj):
    if isinstance(obj, np.generic):   return obj.item()
    elif isinstance(obj, np.ndarray): return obj.tolist()
    elif isinstance(obj, list):       return [convert_numpy(x) for x in obj]
    elif isinstance(obj, dict):       return {k: convert_numpy(v) for k, v in obj.items()}
    return obj


def generate_dataset(num_samples, output_dir, split):
    annotations = {}
    for i in tqdm(range(num_samples), desc=split):
        img, data_coords, pixel_coords, params = generate_scatter_plot(
            image_size=config.IMAGE_SIZE,
            min_points=config.MIN_POINTS,
            max_points=config.MAX_POINTS,
            axis_range=config.AXIS_RANGE,
            min_distance=config.MIN_DISTANCE,
            overlap_probability=config.OVERLAP_PROBABILITY
        )
        H, W     = config.IMAGE_SIZE, config.IMAGE_SIZE
        heatmap  = generate_heatmap((H, W), pixel_coords, sigma=config.GAUSSIAN_SIGMA)

        img_filename = f"{split}_{i:05d}.png"
        img_path     = f"{output_dir}/images/{img_filename}"
        heatmap_path = f"{output_dir}/heatmaps/{img_filename}"

        save_image(img, img_path)
        save_image(heatmap, heatmap_path)

        annotations[img_filename] = {
            "num_points":     len(pixel_coords),
            "pixel_coords":   pixel_coords,
            "image_size":     [H, W],
            "params":         params,
            "data_coords":    data_coords,
            "gaussian_sigma": config.GAUSSIAN_SIGMA,
            "axis_range":     config.AXIS_RANGE,
        }

    metadata_path = f"/scratch/gssodhi/data_extract/metadata/{split}_annotations.json"
    os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
    with open(metadata_path, 'w') as f:
        json.dump(convert_numpy(annotations), f, indent=2)

    print(f"Generated {num_samples} {split} samples")
    print(f"Saved to {output_dir}")
    print(f"Metadata saved to {metadata_path}")


def save_image(img, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if isinstance(img, Image.Image):
        img.save(path)
    elif isinstance(img, np.ndarray):
        Image.fromarray((img * 255).astype(np.uint8), mode='L').save(path)
    else:
        raise ValueError(f"Unknown image type: {type(img)}")


def main():
    np.random.seed(42)
    print("Generating training data...")
    generate_dataset(config.NUM_TRAIN, "/scratch/gssodhi/data_extract/train", "train")
    print("Generating validation data...")
    generate_dataset(config.NUM_VAL,   "/scratch/gssodhi/data_extract/val",   "val")
    
main()