# Generation parameters
NUM_TRAIN = 10000
NUM_VAL = 2000

IMAGE_SIZE = 224  # Height and width

# Point generation
MIN_POINTS = 10
MAX_POINTS = 300

# Overlap settings
OVERLAP_PROBABILITY = 0.2  # 30% chance of intentional overlaps
MIN_DISTANCE = 8  # Minimum distance between points (to control overlap)

# Heatmap generation
GAUSSIAN_SIGMA = 3  # Sigma for Gaussian blobs

# Plot aesthetics
BACKGROUND_COLOR = 'white'
SHOW_AXES = True
SHOW_GRID = True
AXIS_RANGE = (0, 100)  # Data coordinate range
