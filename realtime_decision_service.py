
import asyncio
import json
import joblib
import numpy as np
import pandas as pd
import redis
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.date import DateTrigger
import uvicorn
import logging
from config import *

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="AI Customer Personalization API",
    description="Real-time customer personalization and cart abandonment recovery",
    version="1.0.0"
)

# Initialize Redis client
redis_client = redis.from_url(REDIS_URL, decode_responses=True)

# Initialize scheduler
scheduler = AsyncIOScheduler()
scheduler.start()

# Global variables for models
conversion_model = None
voucher_response_model = None
scaler = None
label_encoders = None

# Pydantic models
class Event(BaseModel):
    event_type: str
    user_id: str
    session_id: str
    timestamp: str
    product_id: Optional[str] = None
    price: Optional[float] = None
    quantity: Optional[int] = None
    page_url: Optional[str] = None
    referrer: Optional[str] = None
    device: Optional[str] = None
    country: Optional[str] = None
    user_segment: Optional[str] = None

class VoucherDecision(BaseModel):
    user_id: str
    session_id: str
    cart_value: float
    voucher_value: float
    confidence: float
    reason: str
    sent_at: str

class SessionState(BaseModel):
    session_id: str
    user_id: str
    events: List[Dict]
    cart_value: float
    cart_items: List[Dict]
    last_activity: str
    status: str  # active, abandoned, converted

# Load ML models
def load_models():
    """Load trained ML models"""
    global conversion_model, voucher_response_model, scaler, label_encoders
    
    try:
        conversion_model = joblib.load(MODEL_CONVERSION_PATH)
        voucher_response_model = joblib.load(MODEL_VOUCHER_RESPONSE_PATH)
        scaler = joblib.load('models/scaler.joblib')
        label_encoders = joblib.load('models/label_encoders.joblib')
        logger.info("✅ ML models loaded successfully")
    except Exception as e:
        logger.error(f"❌ Failed to load models: {e}")
        # Create dummy models for testing
        conversion_model = None
        voucher_response_model = None
        scaler = None
        label_encoders = None

# Session management functions
def get_session_key(session_id: str) -> str:
    """Get Redis key for session"""
    return f"session:{session_id}"

def get_user_key(user_id: str) -> str:
    """Get Redis key for user data"""
    return f"user:{user_id}"

def update_session_state(session_id: str, event: Event):
    """Update session state in Redis"""
    session_key = get_session_key(session_id)
    
    # Get current session data
    session_data = redis_client.hgetall(session_key)
    
    if not session_data:
        # Create new session
        session_data = {
            'user_id': event.user_id,
            'session_start': event.timestamp,
            'last_activity': event.timestamp,
            'event_count': '0',
            'cart_value': '0',
            'cart_items': '[]',
            'status': 'active',
            'page_views': '0',
            'product_views': '0',
            'add_to_cart_events': '0',
            'unique_products': '[]',
            'unique_categories': '[]',
            'device': event.device or 'unknown',
            'country': event.country or 'unknown',
            'user_segment': event.user_segment or 'unknown'
        }
    
    # Update session data
    session_data['last_activity'] = event.timestamp
    session_data['event_count'] = str(int(session_data.get('event_count', 0)) + 1)
    
    # Update event-specific counters
    if event.event_type == 'page_view':
        session_data['page_views'] = str(int(session_data.get('page_views', 0)) + 1)
    elif event.event_type == 'product_view':
        session_data['product_views'] = str(int(session_data.get('product_views', 0)) + 1)
        # Update unique products and categories
        unique_products = json.loads(session_data.get('unique_products', '[]'))
        unique_categories = json.loads(session_data.get('unique_categories', '[]'))
        
        if event.product_id and event.product_id not in unique_products:
            unique_products.append(event.product_id)
        if hasattr(event, 'product_category') and event.product_category not in unique_categories:
            unique_categories.append(event.product_category)
        
        session_data['unique_products'] = json.dumps(unique_products)
        session_data['unique_categories'] = json.dumps(unique_categories)
    
    elif event.event_type == 'add_to_cart':
        session_data['add_to_cart_events'] = str(int(session_data.get('add_to_cart_events', 0)) + 1)
        
        # Update cart
        cart_items = json.loads(session_data.get('cart_items', '[]'))
        cart_value = float(session_data.get('cart_value', 0))
        
        if event.product_id and event.price and event.quantity:
            cart_items.append({
                'product_id': event.product_id,
                'price': event.price,
                'quantity': event.quantity
            })
            cart_value += event.price * event.quantity
        
        session_data['cart_items'] = json.dumps(cart_items)
        session_data['cart_value'] = str(cart_value)
    
    elif event.event_type == 'purchase':
        session_data['status'] = 'converted'
    
    # Save updated session data
    redis_client.hset(session_key, mapping=session_data)
    
    # Set expiration (24 hours)
    redis_client.expire(session_key, 86400)

def compute_session_features(session_id: str) -> np.ndarray:
    """Compute features for ML model from session data"""
    session_key = get_session_key(session_id)
    session_data = redis_client.hgetall(session_key)
    
    if not session_data:
        return None
    
    # Extract features
    features = {
        'max_cart_value': float(session_data.get('cart_value', 0)),
        'event_count': int(session_data.get('event_count', 0)),
        'session_duration_minutes': 0,  # Would need to calculate from timestamps
        'recency_days': 30,  # Default value, would need user history
        'frequency': 1,  # Default value
        'monetary_value': 0,  # Default value
        'avg_order_value': 0,  # Default value
        'page_views': int(session_data.get('page_views', 0)),
        'product_views': int(session_data.get('product_views', 0)),
        'add_to_cart_events': int(session_data.get('add_to_cart_events', 0)),
        'unique_products_viewed': len(json.loads(session_data.get('unique_products', '[]'))),
        'unique_categories_viewed': len(json.loads(session_data.get('unique_categories', '[]'))),
        'avg_product_price_viewed': 0,  # Would need to track this
        'bounce_rate': 1 if int(session_data.get('event_count', 0)) == 1 else 0,
        'hour_of_day': datetime.now().hour,
        'day_of_week': datetime.now().weekday(),
        'is_weekend': 1 if datetime.now().weekday() in [5, 6] else 0,
        'is_business_hours': 1 if 9 <= datetime.now().hour <= 17 else 0,
        'cart_items_count': len(json.loads(session_data.get('cart_items', '[]'))),
        'avg_cart_item_price': 0,  # Would need to calculate
        'cart_categories_count': 0  # Would need to track
    }
    
    # Convert to numpy array
    feature_values = list(features.values())
    return np.array(feature_values).reshape(1, -1)

def predict_conversion_probability(session_id: str) -> float:
    """Predict conversion probability for a session"""
    if not conversion_model or not scaler:
        # Return default probability based on cart value
        session_key = get_session_key(session_id)
        session_data = redis_client.hgetall(session_key)
        cart_value = float(session_data.get('cart_value', 0))
        return min(0.3, cart_value / 1000)  # Simple heuristic
    
    features = compute_session_features(session_id)
    if features is None:
        return 0.1
    
    # Scale features
    features_scaled = scaler.transform(features)
    
    # Predict
    if hasattr(conversion_model, 'predict_proba'):
        prob = conversion_model.predict_proba(features_scaled)[0, 1]
    else:
        prob = conversion_model.predict(features_scaled)[0]
    
    return float(prob)

def predict_voucher_response(session_id: str, voucher_value: float) -> float:
    """Predict voucher response probability"""
    if not voucher_response_model:
        # Return default response rate
        return 0.15
    
    features = compute_session_features(session_id)
    if features is None:
        return 0.1
    
    # Add voucher value as feature
    features_with_voucher = np.append(features, [[voucher_value]], axis=1)
    
    # Scale features
    features_scaled = scaler.transform(features_with_voucher)
    
    # Predict
    if hasattr(voucher_response_model, 'predict_proba'):
        prob = voucher_response_model.predict_proba(features_scaled)[0, 1]
    else:
        prob = voucher_response_model.predict(features_scaled)[0]
    
    return float(prob)

def calculate_expected_value(session_id: str, voucher_value: float) -> float:
    """Calculate expected value of sending a voucher"""
    session_key = get_session_key(session_id)
    session_data = redis_client.hgetall(session_key)
    
    cart_value = float(session_data.get('cart_value', 0))
    
    # Predict probabilities
    p_no_voucher = predict_conversion_probability(session_id)
    p_with_voucher = predict_voucher_response(session_id, voucher_value)
    
    # Calculate expected value
    # Expected gain = (P(with voucher) - P(no voucher)) * cart_value - P(with voucher) * voucher_value
    expected_gain = (p_with_voucher - p_no_voucher) * cart_value - p_with_voucher * voucher_value
    
    return expected_gain

def determine_voucher_value(session_id: str) -> Optional[float]:
    """Determine optimal voucher value for a session"""
    session_key = get_session_key(session_id)
    session_data = redis_client.hgetall(session_key)
    
    cart_value = float(session_data.get('cart_value', 0))
    user_segment = session_data.get('user_segment', 'unknown')
    
    # Define voucher candidates
    voucher_candidates = [5, 10, 15, 20, 25, 30]
    
    best_voucher = None
    best_expected_value = -float('inf')
    
    for voucher_value in voucher_candidates:
        if voucher_value > cart_value * 0.5:  # Don't exceed 50% of cart value
            continue
        
        expected_value = calculate_expected_value(session_id, voucher_value)
        
        if expected_value > best_expected_value and expected_value > 0:
            best_expected_value = expected_value
            best_voucher = voucher_value
    
    return best_voucher

async def send_voucher(user_id: str, session_id: str, voucher_value: float, reason: str):
    """Send voucher to user (simulated)"""
    voucher_decision = VoucherDecision(
        user_id=user_id,
        session_id=session_id,
        cart_value=float(redis_client.hget(get_session_key(session_id), 'cart_value') or 0),
        voucher_value=voucher_value,
        confidence=0.8,  # Would calculate from model confidence
        reason=reason,
        sent_at=datetime.now().isoformat()
    )
    
    # Store voucher decision
    redis_client.lpush('voucher_decisions', voucher_decision.json())
    
    # Log the action
    logger.info(f"🎫 VOUCHER SENT: User {user_id}, Session {session_id}, Value ${voucher_value}, Reason: {reason}")
    
    # In production, this would trigger email/SMS/push notification
    # For now, we'll just log it
    print(f"📧 EMAIL: Dear customer, here's your ${voucher_value} discount code: SAVE{voucher_value}")
    print(f"📱 SMS: Get ${voucher_value} off your order! Use code SAVE{voucher_value}")

async def check_abandonment(session_id: str):
    """Check if session should be considered abandoned and send voucher if appropriate"""
    session_key = get_session_key(session_id)
    session_data = redis_client.hgetall(session_key)
    
    if not session_data:
        return
    
    # Check if already converted or already sent voucher
    if session_data.get('status') == 'converted':
        return
    
    if redis_client.hexists(session_key, 'voucher_sent'):
        return
    
    # Check if session has cart activity
    cart_value = float(session_data.get('cart_value', 0))
    if cart_value < VOUCHER_MIN_CART_VALUE:
        return
    
    # Check if enough time has passed since last activity
    last_activity = datetime.fromisoformat(session_data.get('last_activity', datetime.now().isoformat()))
    time_since_activity = datetime.now() - last_activity
    
    if time_since_activity.total_seconds() < ABANDONMENT_TIMEOUT_MINUTES * 60:
        return
    
    # Mark as abandoned
    redis_client.hset(session_key, 'status', 'abandoned')
    
    # Determine if we should send a voucher
    user_id = session_data.get('user_id')
    user_segment = session_data.get('user_segment', 'unknown')
    
    # Simple policy: send voucher based on user segment and cart value
    should_send_voucher = False
    reason = ""
    
    if user_segment == 'high_value' and cart_value > 100:
        should_send_voucher = True
        reason = "High-value customer with significant cart value"
    elif user_segment == 'regular' and cart_value > 75:
        should_send_voucher = True
        reason = "Regular customer with good cart value"
    elif cart_value > 150:  # High cart value regardless of segment
        should_send_voucher = True
        reason = "High cart value customer"
    
    if should_send_voucher:
        # Determine voucher value
        voucher_value = determine_voucher_value(session_id)
        
        if voucher_value:
            # Send voucher
            await send_voucher(user_id, session_id, voucher_value, reason)
            
            # Mark voucher as sent
            redis_client.hset(session_key, 'voucher_sent', 'true')
            redis_client.hset(session_key, 'voucher_value', str(voucher_value))

# API Endpoints
@app.on_event("startup")
async def startup_event():
    """Initialize the application"""
    load_models()
    logger.info("🚀 AI Customer Personalization API started")

@app.post("/events")
async def track_event(event: Event, background_tasks: BackgroundTasks):
    """Track user events"""
    try:
        # Update session state
        update_session_state(event.session_id, event)
        
        # Schedule abandonment check for cart events
        if event.event_type == 'add_to_cart':
            # Schedule abandonment check
            check_time = datetime.now() + timedelta(minutes=ABANDONMENT_TIMEOUT_MINUTES)
            scheduler.add_job(
                check_abandonment,
                DateTrigger(run_date=check_time),
                args=[event.session_id],
                id=f"abandonment_check_{event.session_id}"
            )
        
        return {"status": "success", "message": "Event tracked successfully"}
    
    except Exception as e:
        logger.error(f"Error tracking event: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/session/{session_id}")
async def get_session_state(session_id: str):
    """Get current session state"""
    session_key = get_session_key(session_id)
    session_data = redis_client.hgetall(session_key)
    
    if not session_data:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return {
        "session_id": session_id,
        "user_id": session_data.get('user_id'),
        "status": session_data.get('status'),
        "cart_value": float(session_data.get('cart_value', 0)),
        "cart_items": json.loads(session_data.get('cart_items', '[]')),
        "last_activity": session_data.get('last_activity'),
        "event_count": int(session_data.get('event_count', 0)),
        "voucher_sent": redis_client.hexists(session_key, 'voucher_sent')
    }

@app.get("/predict/conversion/{session_id}")
async def predict_conversion(session_id: str):
    """Predict conversion probability for a session"""
    try:
        probability = predict_conversion_probability(session_id)
        return {
            "session_id": session_id,
            "conversion_probability": probability,
            "confidence": "high" if probability > 0.7 else "medium" if probability > 0.3 else "low"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/voucher/decide/{session_id}")
async def decide_voucher(session_id: str):
    """Manually trigger voucher decision for a session"""
    try:
        await check_abandonment(session_id)
        return {"status": "success", "message": "Voucher decision processed"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/vouchers/sent")
async def get_sent_vouchers(limit: int = 100):
    """Get list of sent vouchers"""
    try:
        vouchers = redis_client.lrange('voucher_decisions', 0, limit - 1)
        return [json.loads(voucher) for voucher in vouchers]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analytics/dashboard")
async def get_analytics_dashboard():
    """Get analytics dashboard data"""
    try:
        # Get all session keys
        session_keys = redis_client.keys("session:*")
        
        total_sessions = len(session_keys)
        active_sessions = 0
        abandoned_sessions = 0
        converted_sessions = 0
        total_cart_value = 0
        vouchers_sent = 0
        
        for key in session_keys:
            session_data = redis_client.hgetall(key)
            status = session_data.get('status', 'active')
            
            if status == 'active':
                active_sessions += 1
            elif status == 'abandoned':
                abandoned_sessions += 1
            elif status == 'converted':
                converted_sessions += 1
            
            total_cart_value += float(session_data.get('cart_value', 0))
            
            if redis_client.hexists(key, 'voucher_sent'):
                vouchers_sent += 1
        
        return {
            "total_sessions": total_sessions,
            "active_sessions": active_sessions,
            "abandoned_sessions": abandoned_sessions,
            "converted_sessions": converted_sessions,
            "conversion_rate": converted_sessions / total_sessions if total_sessions > 0 else 0,
            "abandonment_rate": abandoned_sessions / total_sessions if total_sessions > 0 else 0,
            "total_cart_value": total_cart_value,
            "vouchers_sent": vouchers_sent,
            "voucher_rate": vouchers_sent / total_sessions if total_sessions > 0 else 0
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": conversion_model is not None and voucher_response_model is not None,
        "redis_connected": redis_client.ping()
    }

if __name__ == "__main__":
    uvicorn.run(
        "3_realtime_decision_service:app",
        host=API_HOST,
        port=API_PORT,
        reload=True,
        log_level="info"
    )
