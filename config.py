"""
Centraliseerde configuratie voor Smart Health Dashboard
Alle instellingen en paden op één plek
"""

from pathlib import Path
import os
import logging

from dotenv import load_dotenv

# ──────────────────────────────────────────────────────────────────────────────
# BASE PATHS
# ──────────────────────────────────────────────────────────────────────────────

CODE_DIR = Path(__file__).resolve().parent
BASE = CODE_DIR.parent if os.getenv('IS_STREAMLIT') else CODE_DIR
HTML_DIR = BASE / "HTML"
TABLES_DIR = BASE / "Tabellen"
DATA_DIR = CODE_DIR / "data"

# Load .env before any environment-dependent values are read
load_dotenv(dotenv_path=CODE_DIR / ".env", override=True)

# ──────────────────────────────────────────────────────────────────────────────
# DATA SOURCE
# ──────────────────────────────────────────────────────────────────────────────

SYNTHETIC_DB_PATH = CODE_DIR / "synthetic_health.db"
DEFAULT_DB_URL = f"sqlite:///{SYNTHETIC_DB_PATH}"
DATA_SOURCE = os.getenv('DATA_SOURCE', 'csv').strip().lower()
CSV_DATA_DIR = Path(os.getenv('CSV_DATA_DIR', DATA_DIR))

env_db = os.getenv('DATABASE_URL')
if env_db and ('sqlite' in env_db or env_db.startswith('sqlite')):
    DB_URL = env_db
else:
    DB_URL = DEFAULT_DB_URL

DB_URL_POSTGRES = DB_URL
DB_URL_SMART_HEALTH = DB_URL

# ──────────────────────────────────────────────────────────────────────────────
# AUTHENTICATION / SECURITY
# ──────────────────────────────────────────────────────────────────────────────

AUTH_ENABLED_ENV = "SMART_HEALTH_AUTH_ENABLED"
AUTH_CONFIG_FILE = Path(os.getenv('AUTH_CONFIG_FILE', CODE_DIR / 'auth_config.yaml'))
AUTH_COOKIE_KEY = os.getenv('AUTH_COOKIE_KEY', os.getenv('STREAMLIT_COOKIE_KEY', ""))
AUTH_COOKIE_EXPIRY_DAYS = float(os.getenv('AUTH_COOKIE_EXPIRY_DAYS', "1"))
AUTH_SESSION_TIMEOUT_MINUTES = int(os.getenv('AUTH_SESSION_TIMEOUT_MINUTES', "15"))
AUTH_RATE_LIMIT_MAX_ATTEMPTS = int(os.getenv('AUTH_RATE_LIMIT_MAX_ATTEMPTS', "5"))
AUTH_RATE_LIMIT_PERIOD_MINUTES = int(os.getenv('AUTH_RATE_LIMIT_PERIOD_MINUTES', "10"))

LOGS_DIR = CODE_DIR / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
AUTH_LOG_FILE = LOGS_DIR / "auth.log"

# Optie 2: Lokale development (uncomment if needed)
# DB_URL = "mysql+pymysql://user:password@localhost/smart_health"

# ──────────────────────────────────────────────────────────────────────────────
# STREAMLIT SETTINGS
# ──────────────────────────────────────────────────────────────────────────────

STREAMLIT_CONFIG = {
    'page_title': 'Smart Health Dashboard',
    'page_icon': '🏥',
    'layout': 'wide',
    'initial_sidebar_state': 'expanded',
}

# ──────────────────────────────────────────────────────────────────────────────
# LOGGING
# ──────────────────────────────────────────────────────────────────────────────

LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format=LOG_FORMAT,
)

logger = logging.getLogger(__name__)

def setup_logger(name: str, level=None):
    """Setup logger voor modules."""
    return logging.getLogger(name)

# ──────────────────────────────────────────────────────────────────────────────
# FEATURE FLAGS
# ──────────────────────────────────────────────────────────────────────────────

FEATURES = {
    'ml_models': True,              # ML modellen tonen
    'longitudinal_analysis': True,  # Longitudinale analyse
    'maps': True,                   # Geografische kaarten
    'per_opdrachtgever': True,      # Per opdrachtgever analyses
    'debug_mode': os.getenv('DEBUG', 'false').lower() == 'true',
}

# ──────────────────────────────────────────────────────────────────────────────
# CACHE SETTINGS
# ──────────────────────────────────────────────────────────────────────────────

CACHE_TTL_SECONDS = {
    'main_data': 3600,              # 1 uur
    'longitudinal_data': 7200,      # 2 uur
    'statistics': 1800,             # 30 min
    'plots': 1800,                  # 30 min
}

# ──────────────────────────────────────────────────────────────────────────────
# DATA PROCESSING
# ──────────────────────────────────────────────────────────────────────────────

# Minimum aantal participants voor weergave
MIN_PARTICIPANTS = {
    'store_analysis': 2,
    'longitudinal': 10,
    'statistical_test': 10,
}

# Outlier detection
OUTLIER_DETECTION = {
    'method': 'iqr',  # 'iqr', 'zscore', 'isolation_forest'
    'iqr_multiplier': 1.5,
    'zscore_threshold': 3.0,
}

# ──────────────────────────────────────────────────────────────────────────────
# VALIDATION & DEFAULTS
# ──────────────────────────────────────────────────────────────────────────────

def validate_config():
    """Valideer de geconfigureerde demo-databron."""
    if DATA_SOURCE == 'csv':
        if not CSV_DATA_DIR.exists():
            logger.error("CSV data directory ontbreekt: %s", CSV_DATA_DIR)
            return False
        logger.info("CSV data source configured: %s", CSV_DATA_DIR)
        return True
    if not DB_URL:
        logger.error("❌ DATABASE_URL not configured. Set DATABASE_URL environment variable.")
        return False
    
    logger.info("✓ Database URL configured for direct data loading")
    return True

# Auto-validate on import
if __name__ != '__main__':
    try:
        validate_config()
    except Exception as e:
        logger.warning(f"Config validation warning: {e}")
