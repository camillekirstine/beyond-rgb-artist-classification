from pathlib import Path

# --------------------------------------------------
# Project directories
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

DATA_ARCHIVE = DATA_RAW_DIR / "paintingsSMALL.imscc"
DATA_DIR = DATA_PROCESSED_DIR / "paintingsSMALL"

NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

SRC_DIR = PROJECT_ROOT / "src"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = OUTPUTS_DIR / "models"
METRICS_DIR = OUTPUTS_DIR / "metrics"
FIGURES_DIR = OUTPUTS_DIR / "figures"
TABLES_DIR = OUTPUTS_DIR / "tables"
LOGS_DIR = OUTPUTS_DIR / "logs"

REPORTS_DIR = PROJECT_ROOT / "reports"
REPORT_FIGURES_DIR = REPORTS_DIR / "figures"
REPORT_TABLES_DIR = REPORTS_DIR / "tables"
REPORT_DRAFT_DIR = REPORTS_DIR / "draft"


# --------------------------------------------------
# Dataset configuration
# --------------------------------------------------

CLASSES = ["monet", "picasso", "vangogh"]

NUM_CLASSES = len(CLASSES)

IMAGE_SIZE = (180, 180)

IMAGE_CHANNELS = 3

BATCH_SIZE = 16


# --------------------------------------------------
# Experiment configuration
# --------------------------------------------------

DEFAULT_SEED = 42

MAX_EPOCHS = 30

LEARNING_RATE = 0.001

EARLY_STOPPING_PATIENCE = 6

REDUCE_LR_PATIENCE = 3

REDUCE_LR_FACTOR = 0.5

MIN_LEARNING_RATE = 0.00001

DROPOUT_RATE = 0.4


# --------------------------------------------------
# Experimental conditions
# --------------------------------------------------

COLOR_SPACES = ["rgb", "hsv", "grayscale"]
