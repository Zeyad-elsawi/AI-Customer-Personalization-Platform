import os
from dotenv import load_dotenv

load_dotenv()

# Redis Configuration
REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379/0')

# Abandonment Detection
ABANDONMENT_TIMEOUT_MINUTES = int(os.getenv('ABANDONMENT_TIMEOUT_MINUTES', 20))

# Voucher Policy
VOUCHER_MIN_CART_VALUE = float(os.getenv('VOUCHER_MIN_CART_VALUE', 50))
VOUCHER_MAX_DISCOUNT_PERCENT = float(os.getenv('VOUCHER_MAX_DISCOUNT_PERCENT', 20))

# API Configuration
API_HOST = os.getenv('API_HOST', '0.0.0.0')
API_PORT = int(os.getenv('API_PORT', 8000))

# Model Configuration
MODEL_CONVERSION_PATH = 'models/conversion_model.joblib'
MODEL_VOUCHER_RESPONSE_PATH = 'models/voucher_response_model.joblib'
