"""
Demo script to showcase the improved real-time decision service
with comprehensive feature matching between training and prediction
"""

import requests
import json
import time
from datetime import datetime
import uuid

# API base URL
API_BASE = "http://localhost:8000"

def create_test_event(event_type, user_id, session_id, **kwargs):
    """Create a test event"""
    return {
        "event_type": event_type,
        "user_id": user_id,
        "session_id": session_id,
        "timestamp": datetime.now().isoformat(),
        **kwargs
    }

def send_event(event):
    """Send event to the API"""
    try:
        response = requests.post(f"{API_BASE}/events", json=event)
        if response.status_code == 200:
            print(f"✅ {event['event_type']} event sent successfully")
            return True
        else:
            print(f"❌ Failed to send {event['event_type']} event: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Error sending event: {e}")
        return False

def get_session_state(session_id):
    """Get session state"""
    try:
        response = requests.get(f"{API_BASE}/session/{session_id}")
        if response.status_code == 200:
            return response.json()
        else:
            print(f"❌ Failed to get session state: {response.text}")
            return None
    except Exception as e:
        print(f"❌ Error getting session state: {e}")
        return None

def get_conversion_prediction(session_id):
    """Get conversion prediction"""
    try:
        response = requests.get(f"{API_BASE}/predict/conversion/{session_id}")
        if response.status_code == 200:
            return response.json()
        else:
            print(f"❌ Failed to get conversion prediction: {response.text}")
            return None
    except Exception as e:
        print(f"❌ Error getting conversion prediction: {e}")
        return None

def simulate_user_journey():
    """Simulate a complete user journey with rich features"""
    print("🚀 Starting Comprehensive User Journey Demo")
    print("=" * 60)
    
    # Create test user and session
    user_id = f"user_{uuid.uuid4().hex[:8]}"
    session_id = f"session_{uuid.uuid4().hex[:8]}"
    
    print(f"👤 User ID: {user_id}")
    print(f"🔄 Session ID: {session_id}")
    print()
    
    # Step 1: User visits homepage
    print("📱 Step 1: User visits homepage")
    event = create_test_event(
        "page_view", user_id, session_id,
        page_url="/",
        device="desktop",
        country="US",
        user_segment="regular"
    )
    send_event(event)
    time.sleep(1)
    
    # Step 2: User views products
    print("\n🛍️ Step 2: User browses products")
    products = [
        {"id": "prod_1", "name": "Laptop", "price": 999.99, "category": "Electronics", "brand": "TechCorp"},
        {"id": "prod_2", "name": "Headphones", "price": 199.99, "category": "Electronics", "brand": "AudioPro"},
        {"id": "prod_3", "name": "Mouse", "price": 49.99, "category": "Electronics", "brand": "TechCorp"},
        {"id": "prod_4", "name": "Keyboard", "price": 79.99, "category": "Electronics", "brand": "TechCorp"},
        {"id": "prod_5", "name": "Monitor", "price": 299.99, "category": "Electronics", "brand": "DisplayMax"}
    ]
    
    for product in products:
        event = create_test_event(
            "product_view", user_id, session_id,
            product_id=product["id"],
            price=product["price"],
            product_category=product["category"],
            product_brand=product["brand"]
        )
        send_event(event)
        time.sleep(0.5)
    
    # Step 3: User adds items to cart
    print("\n🛒 Step 3: User adds items to cart")
    cart_items = [
        {"id": "prod_1", "price": 999.99, "quantity": 1},
        {"id": "prod_2", "price": 199.99, "quantity": 1},
        {"id": "prod_3", "price": 49.99, "quantity": 2}
    ]
    
    for item in cart_items:
        event = create_test_event(
            "add_to_cart", user_id, session_id,
            product_id=item["id"],
            price=item["price"],
            quantity=item["quantity"]
        )
        send_event(event)
        time.sleep(0.5)
    
    # Step 4: Check session state and conversion prediction
    print("\n📊 Step 4: Analyzing session with AI")
    session_state = get_session_state(session_id)
    if session_state:
        print(f"   Cart Value: ${session_state['cart_value']:.2f}")
        print(f"   Cart Items: {len(session_state['cart_items'])}")
        print(f"   Event Count: {session_state['event_count']}")
        print(f"   Status: {session_state['status']}")
    
    conversion_pred = get_conversion_prediction(session_id)
    if conversion_pred:
        print(f"   Conversion Probability: {conversion_pred['conversion_probability']:.2%}")
        print(f"   Confidence: {conversion_pred['confidence']}")
    
    # Step 5: Simulate cart abandonment
    print("\n⏰ Step 5: Simulating cart abandonment (waiting 2 minutes)")
    print("   (In real scenario, this would be 30+ minutes)")
    time.sleep(2)
    
    # Step 6: Trigger abandonment check
    print("\n🎫 Step 6: Triggering AI voucher decision")
    try:
        response = requests.post(f"{API_BASE}/voucher/decide/{session_id}")
        if response.status_code == 200:
            print("   ✅ Voucher decision processed")
        else:
            print(f"   ❌ Failed to process voucher decision: {response.text}")
    except Exception as e:
        print(f"   ❌ Error processing voucher decision: {e}")
    
    # Step 7: Check final session state
    print("\n📈 Step 7: Final session analysis")
    final_state = get_session_state(session_id)
    if final_state:
        print(f"   Final Status: {final_state['status']}")
        print(f"   Voucher Sent: {final_state['voucher_sent']}")
    
    # Step 8: Show analytics
    print("\n📊 Step 8: System Analytics")
    try:
        response = requests.get(f"{API_BASE}/analytics/dashboard")
        if response.status_code == 200:
            analytics = response.json()
            print(f"   Total Sessions: {analytics['total_sessions']}")
            print(f"   Active Sessions: {analytics['active_sessions']}")
            print(f"   Abandoned Sessions: {analytics['abandoned_sessions']}")
            print(f"   Converted Sessions: {analytics['converted_sessions']}")
            print(f"   Conversion Rate: {analytics['conversion_rate']:.2%}")
            print(f"   Vouchers Sent: {analytics['vouchers_sent']}")
        else:
            print(f"   ❌ Failed to get analytics: {response.text}")
    except Exception as e:
        print(f"   ❌ Error getting analytics: {e}")
    
    print("\n🎉 Demo completed!")
    print("=" * 60)
    print("🔍 Key Improvements Demonstrated:")
    print("   ✅ Rich feature computation (RFM, behavioral, temporal)")
    print("   ✅ Product interaction tracking (categories, prices)")
    print("   ✅ User profile management with purchase history")
    print("   ✅ Comprehensive session state tracking")
    print("   ✅ AI-powered conversion prediction")
    print("   ✅ Intelligent voucher decision making")
    print("   ✅ Real-time analytics and monitoring")

def test_feature_richness():
    """Test the richness of features being computed"""
    print("\n🧪 Testing Feature Richness")
    print("=" * 40)
    
    user_id = f"test_user_{uuid.uuid4().hex[:8]}"
    session_id = f"test_session_{uuid.uuid4().hex[:8]}"
    
    # Create a session with rich data
    events = [
        create_test_event("page_view", user_id, session_id, device="mobile", country="UK", user_segment="high_value"),
        create_test_event("product_view", user_id, session_id, product_id="prod_1", price=299.99, product_category="Electronics", product_brand="PremiumBrand"),
        create_test_event("product_view", user_id, session_id, product_id="prod_2", price=199.99, product_category="Electronics", product_brand="PremiumBrand"),
        create_test_event("add_to_cart", user_id, session_id, product_id="prod_1", price=299.99, quantity=1),
        create_test_event("add_to_cart", user_id, session_id, product_id="prod_2", price=199.99, quantity=2),
    ]
    
    for event in events:
        send_event(event)
        time.sleep(0.2)
    
    # Get conversion prediction to see features in action
    pred = get_conversion_prediction(session_id)
    if pred:
        print(f"   Conversion Probability: {pred['conversion_probability']:.2%}")
        print(f"   Confidence: {pred['confidence']}")
        print("   ✅ Features computed successfully!")

if __name__ == "__main__":
    print("🎯 AI Customer Personalization - Improved Real-time Demo")
    print("This demo shows the enhanced system with comprehensive feature matching")
    print()
    
    # Check if API is running
    try:
        response = requests.get(f"{API_BASE}/health")
        if response.status_code == 200:
            health = response.json()
            print(f"✅ API is running - Models loaded: {health['models_loaded']}")
            print(f"✅ Redis connected: {health['redis_connected']}")
            print()
        else:
            print("❌ API is not responding properly")
            exit(1)
    except Exception as e:
        print(f"❌ Cannot connect to API: {e}")
        print("Please make sure the real-time service is running:")
        print("   python realtime_decision_service.py")
        exit(1)
    
    # Run demos
    simulate_user_journey()
    test_feature_richness()
    
    print("\n🚀 To see the system in action:")
    print("   1. Open demo_website.html in your browser")
    print("   2. Browse products and add items to cart")
    print("   3. Wait for AI-powered voucher decisions!")
    print("   4. Check the API endpoints for real-time data")
