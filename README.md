# 🎯 AI Customer Personalization Platform

A comprehensive AI-powered customer personalization platform that tracks user behavior, detects cart abandonment, and automatically sends personalized vouchers using machine learning - similar to MoEngage.

## 🌟 Features

- **Real-time User Tracking**: Track user behavior across website sessions
- **Cart Abandonment Detection**: Automatically detect when users abandon their shopping carts
- **AI-Powered Voucher System**: Send personalized vouchers based on ML predictions
- **Conversion Prediction**: Predict likelihood of user conversion
- **Behavioral Analytics**: Comprehensive user behavior analysis
- **Real-time Decision Engine**: FastAPI-based real-time decision making
- **Frontend Integration**: Easy-to-integrate JavaScript tracking library

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   FastAPI       │    │   Redis         │
│   Tracking      │───▶│   Decision      │───▶│   Session       │
│   (JavaScript)  │    │   Service       │    │   Store         │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │   ML Models     │
                       │   (LightGBM,    │
                       │   Random Forest)│
                       └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Redis server
- Modern web browser

### Installation

1. **Clone and setup**:
```bash
git clone <repository>
cd ai-customer-personalization
python setup.py
```

2. **Start Redis**:
```bash
# Using Docker (recommended)
docker run -d -p 6379:6379 redis:alpine

# Or install locally
# Windows: Download from GitHub releases
# macOS: brew install redis && brew services start redis
# Linux: sudo apt-get install redis-server
```

3. **Generate training data**:
```bash
python 1_simulate_events.py
```

4. **Train ML models**:
```bash
python 2_training_pipeline.py
```

5. **Start the API server**:
```bash
python 3_realtime_decision_service.py
```

6. **Open demo website**:
```bash
# Open demo_website.html in your browser
# Or serve it with a local server
python -m http.server 8001
```

### Using Docker

```bash
docker-compose up --build
```

## 📁 Project Structure

```
ai-customer-personalization/
├── 1_simulate_events.py          # Synthetic data generator
├── 2_training_pipeline.py        # ML model training
├── 3_realtime_decision_service.py # FastAPI decision service
├── 4_frontend_tracking.js        # JavaScript tracking library
├── demo_website.html             # Demo e-commerce website
├── config.py                     # Configuration settings
├── requirements.txt              # Python dependencies
├── setup.py                      # Setup script
├── docker-compose.yml            # Docker configuration
├── Dockerfile                    # Docker image definition
├── data/                         # Generated data files
│   ├── synthetic_events.parquet
│   └── synthetic_sessions.parquet
└── models/                       # Trained ML models
    ├── conversion_model.joblib
    ├── voucher_response_model.joblib
    └── scaler.joblib
```

## 🔧 Components

### 1. Event Simulation (`1_simulate_events.py`)

Generates synthetic user behavior data including:
- User sessions with realistic browsing patterns
- Product interactions (views, cart additions)
- Purchase behaviors
- Voucher campaigns and responses

**Key Features**:
- Multiple user segments (high-value, regular, occasional)
- Realistic product catalogs
- Behavioral patterns based on user segments
- Voucher campaign simulation

### 2. ML Training Pipeline (`2_training_pipeline.py`)

Trains machine learning models for:
- **Conversion Prediction**: Predicts likelihood of purchase
- **Voucher Response**: Predicts voucher redemption probability

**Models Used**:
- LightGBM for conversion prediction
- Random Forest for voucher response
- Feature engineering with RFM analysis
- Behavioral feature extraction

**Features Engineered**:
- User RFM (Recency, Frequency, Monetary)
- Session-level features
- Behavioral patterns
- Temporal features
- Product interaction features

### 3. Real-time Decision Service (`3_realtime_decision_service.py`)

FastAPI-based service that:
- Tracks user sessions in real-time
- Detects cart abandonment
- Makes voucher decisions using ML models
- Manages session state in Redis

**API Endpoints**:
- `POST /events` - Track user events
- `GET /session/{session_id}` - Get session state
- `GET /predict/conversion/{session_id}` - Predict conversion
- `POST /voucher/decide/{session_id}` - Trigger voucher decision
- `GET /analytics/dashboard` - Analytics dashboard
- `GET /health` - Health check

### 4. Frontend Tracking (`4_frontend_tracking.js`)

JavaScript library for tracking user behavior:
- Automatic event tracking
- E-commerce integration
- Real-time session management
- Cross-platform compatibility

**Tracked Events**:
- Page views
- Product views
- Add to cart
- Remove from cart
- Checkout started
- Purchase
- Scroll depth
- Time on page
- User activity

## 🎯 Usage Examples

### Frontend Integration

```html
<!-- Include the tracking script -->
<script src="4_frontend_tracking.js"></script>

<script>
// Initialize tracker
const tracker = new CustomerPersonalizationTracker({
    apiUrl: 'http://localhost:8000',
    debug: true
});

// Track custom events
tracker.trackAddToCart('SKU_123', 1, 99.99, {
    name: 'Wireless Headphones',
    category: 'Electronics',
    brand: 'TechBrand'
});

// Set user properties
tracker.setUserSegment('high_value');
tracker.setCountry('US');
</script>
```

### API Usage

```python
import requests

# Track an event
event_data = {
    "event_type": "add_to_cart",
    "user_id": "user_123",
    "session_id": "sess_456",
    "timestamp": "2024-01-01T12:00:00Z",
    "product_id": "SKU_123",
    "price": 99.99,
    "quantity": 1
}

response = requests.post('http://localhost:8000/events', json=event_data)

# Get session state
session_response = requests.get('http://localhost:8000/session/sess_456')
print(session_response.json())

# Get analytics
analytics = requests.get('http://localhost:8000/analytics/dashboard')
print(analytics.json())
```

## 📊 Analytics Dashboard

The platform provides comprehensive analytics:

- **Session Metrics**: Total sessions, active sessions, conversion rates
- **Cart Analytics**: Abandonment rates, cart values, recovery rates
- **Voucher Performance**: Send rates, redemption rates, ROI
- **User Segmentation**: Behavior by user segment
- **Real-time Monitoring**: Live session tracking

## 🔧 Configuration

Edit `config.py` to customize:

```python
# Redis Configuration
REDIS_URL = 'redis://localhost:6379/0'

# Abandonment Detection
ABANDONMENT_TIMEOUT_MINUTES = 20

# Voucher Policy
VOUCHER_MIN_CART_VALUE = 50
VOUCHER_MAX_DISCOUNT_PERCENT = 20

# API Configuration
API_HOST = '0.0.0.0'
API_PORT = 8000
```

## 🧪 Testing

### Manual Testing

1. Open `demo_website.html`
2. Browse products and add items to cart
3. Leave the cart without purchasing
4. Wait for abandonment timeout (20 minutes by default)
5. Check API logs for voucher decisions

### API Testing

```bash
# Health check
curl http://localhost:8000/health

# Track event
curl -X POST http://localhost:8000/events \
  -H "Content-Type: application/json" \
  -d '{"event_type":"page_view","user_id":"test_user","session_id":"test_session","timestamp":"2024-01-01T12:00:00Z"}'

# Get analytics
curl http://localhost:8000/analytics/dashboard
```

## 🚀 Production Deployment

### Environment Setup

1. **Redis Cluster**: Use Redis Cluster for high availability
2. **Load Balancer**: Use nginx or similar for API load balancing
3. **Monitoring**: Add Prometheus/Grafana for monitoring
4. **Logging**: Implement structured logging with ELK stack

### Security Considerations

- Implement API authentication
- Use HTTPS for all communications
- Sanitize user inputs
- Implement rate limiting
- Use environment variables for secrets

### Scaling

- **Horizontal Scaling**: Run multiple API instances
- **Database**: Consider PostgreSQL for persistent storage
- **Caching**: Use Redis for session caching
- **Message Queue**: Use Kafka for event streaming

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Inspired by MoEngage's customer personalization platform
- Built with FastAPI, Redis, and scikit-learn
- Uses LightGBM for high-performance ML predictions

## 📞 Support

For questions or support, please open an issue on GitHub or contact the development team.

---

**Built with ❤️ for better customer experiences**
