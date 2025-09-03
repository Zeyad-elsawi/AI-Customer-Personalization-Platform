"""
Simulated Event Generator for AI Customer Personalization Platform
Generates synthetic user sessions, events, and purchase behaviors for training ML models
"""

import pandas as pd
import numpy as np
import random
import uuid
from datetime import datetime, timedelta
import json
from typing import List, Dict, Any

# Configuration
N_USERS = 1000
N_PRODUCTS = 200
DAYS = 90
ABANDONMENT_RATE = 0.3 
VOUCHER_RESPONSE_RATE = 0.15  

# Product catalog
def generate_products(n_products: int) -> List[Dict[str, Any]]:
    """Generate synthetic product catalog"""
    categories = ["Electronics", "Clothing", "Home", "Books", "Sports", "Beauty", "Toys", "Food"]
    brands = ["BrandA", "BrandB", "BrandC", "BrandD", "BrandE"]
    
    products = []
    for i in range(n_products):
        products.append({
            "product_id": f"SKU_{i:04d}",
            "name": f"Product {i}",
            "category": random.choice(categories),
            "brand": random.choice(brands),
            "price": round(random.uniform(5, 500), 2),
            "popularity": random.uniform(0.1, 1.0)  
        })
    return products

# User profiles
def generate_user_profiles(n_users: int) -> List[Dict[str, Any]]:
    
    countries = ["US", "UK", "CA", "AU", "DE", "FR", "IT", "ES"]
    devices = ["desktop", "mobile", "tablet"]
    
    users = []
    for i in range(n_users):
        # Create different user segments
        if i < n_users * 0.1:  # 10% high-value customers
            segment = "high_value"
            activity_rate = random.uniform(0.8, 1.2)
            price_sensitivity = random.uniform(0.1, 0.3)
        elif i < n_users * 0.3:  # 20% regular customers
            segment = "regular"
            activity_rate = random.uniform(0.4, 0.8)
            price_sensitivity = random.uniform(0.3, 0.6)
        else:  # 70% occasional customers
            segment = "occasional"
            activity_rate = random.uniform(0.1, 0.4)
            price_sensitivity = random.uniform(0.6, 0.9)
        
        users.append({
            "user_id": f"user_{i:05d}",
            "country": random.choice(countries),
            "device_preference": random.choice(devices),
            "segment": segment,
            "activity_rate": activity_rate,
            "price_sensitivity": price_sensitivity,
            "lifetime_value": random.uniform(0, 2000),
            "last_purchase_days_ago": random.randint(0, 365)
        })
    return users

def generate_session_events(user: Dict[str, Any], products: List[Dict[str, Any]], 
                          start_date: datetime, session_duration_hours: float = 2) -> List[Dict[str, Any]]:
    
    session_id = str(uuid.uuid4())
    events = []
    cart = []
    session_start = start_date + timedelta(minutes=random.randint(0, 60))
    
    # Determine session length based on user activity rate
    max_events = int(20 * user['activity_rate'])
    n_events = random.randint(1, max_events)
    
    # Generate events over session duration
    for i in range(n_events):
        event_time = session_start + timedelta(
            minutes=random.randint(0, int(session_duration_hours * 60))
        )
        
        # Choose event type based on probabilities
        event_weights = [0.4, 0.25, 0.15, 0.05, 0.05, 0.05, 0.05]  # page_view, product_view, add_to_cart, remove_from_cart, checkout_started, purchase, session_end
        event_type = random.choices(
            ["page_view", "product_view", "add_to_cart", "remove_from_cart", 
             "checkout_started", "purchase", "session_end"],
            weights=event_weights
        )[0]
        
        # Select product (biased by popularity and user preferences)
        product = select_product_for_user(user, products)
        
        if event_type in ["add_to_cart", "remove_from_cart", "purchase"]:
            quantity = random.randint(1, 3)
            if event_type == "add_to_cart":
                cart.append({
                    "product_id": product["product_id"],
                    "price": product["price"],
                    "quantity": quantity
                })
            elif event_type == "remove_from_cart" and cart:
                # Remove random item from cart
                if random.random() < 0.3:  # 30% chance to remove item
                    cart.pop(random.randint(0, len(cart) - 1))
        
        # Calculate current cart value
        cart_value = sum(item["price"] * item["quantity"] for item in cart)
        
        event = {
            "event_id": str(uuid.uuid4()),
            "user_id": user["user_id"],
            "session_id": session_id,
            "timestamp": event_time,
            "event_type": event_type,
            "product_id": product["product_id"],
            "product_category": product["category"],
            "product_brand": product["brand"],
            "price": product["price"],
            "quantity": quantity if event_type in ["add_to_cart", "remove_from_cart", "purchase"] else 0,
            "cart_value": cart_value,
            "cart_items_count": len(cart),
            "page_url": f"/{product['category'].lower()}/{product['product_id']}",
            "referrer": random.choice(["direct", "google", "facebook", "email", "organic"]),
            "device": user["device_preference"],
            "country": user["country"],
            "user_segment": user["segment"]
        }
        
        events.append(event)
        
        # End session if purchase or session_end
        if event_type in ["purchase", "session_end"]:
            break
    
    return events

def select_product_for_user(user: Dict[str, Any], products: List[Dict[str, Any]]) -> Dict[str, Any]:
    
    weights = []
    for product in products:
        # Base weight from product popularity
        weight = product["popularity"]
        
        # Adjust for price sensitivity
        if user["price_sensitivity"] > 0.7 and product["price"] > 100:
            weight *= 0.3  # Price-sensitive users less likely to view expensive items
        elif user["price_sensitivity"] < 0.3 and product["price"] < 50:
            weight *= 0.5  # High-value users less likely to view cheap items
        
        weights.append(weight)
    
    return random.choices(products, weights=weights)[0]

def simulate_voucher_campaigns(events_df: pd.DataFrame) -> pd.DataFrame:
    """Simulate voucher campaigns and their effects"""
    # Find abandoned cart sessions
    abandoned_sessions = events_df[
        (events_df['event_type'] == 'add_to_cart') & 
        (events_df['cart_value'] > 30)  # Minimum cart value for voucher
    ].groupby('session_id').agg({
        'user_id': 'first',
        'cart_value': 'max',
        'timestamp': 'max',
        'user_segment': 'first'
    }).reset_index()
    
    # Randomly select sessions to receive vouchers (simulate A/B test)
    voucher_sessions = abandoned_sessions.sample(
        frac=0.3, random_state=42
    ).copy()
    
    # Determine voucher value based on cart value and user segment
    voucher_sessions['voucher_value'] = voucher_sessions.apply(
        lambda row: min(
            row['cart_value'] * 0.1,  # 10% of cart value
            50  # Max $50 voucher
        ) if row['user_segment'] == 'high_value' else min(
            row['cart_value'] * 0.15,  # 15% for regular users
            25  # Max $25 voucher
        ), axis=1
    )
    
    # Simulate voucher redemption
    voucher_sessions['voucher_redeemed'] = voucher_sessions.apply(
        lambda row: random.random() < VOUCHER_RESPONSE_RATE * (1 + (0.2 if row['user_segment'] == 'high_value' else 0)), 
        axis=1
    )
    
    # Add voucher events to main events dataframe
    voucher_events = []
    for _, row in voucher_sessions.iterrows():
        # Voucher sent event
        voucher_events.append({
            "event_id": str(uuid.uuid4()),
            "user_id": row['user_id'],
            "session_id": row['session_id'],
            "timestamp": row['timestamp'] + timedelta(minutes=random.randint(30, 120)),
            "event_type": "voucher_sent",
            "product_id": None,
            "product_category": None,
            "product_brand": None,
            "price": 0,
            "quantity": 0,
            "cart_value": row['cart_value'],
            "cart_items_count": 0,
            "page_url": None,
            "referrer": "email",
            "device": "email",
            "country": "US",
            "user_segment": row['user_segment'],
            "voucher_value": row['voucher_value']
        })
        
        # Voucher redeemed event (if redeemed)
        if row['voucher_redeemed']:
            voucher_events.append({
                "event_id": str(uuid.uuid4()),
                "user_id": row['user_id'],
                "session_id": str(uuid.uuid4()),  # New session for redemption
                "timestamp": row['timestamp'] + timedelta(days=random.randint(1, 7)),
                "event_type": "voucher_redeemed",
                "product_id": None,
                "product_category": None,
                "product_brand": None,
                "price": 0,
                "quantity": 0,
                "cart_value": row['cart_value'] * 0.8,  # Slightly lower value
                "cart_items_count": 0,
                "page_url": None,
                "referrer": "voucher",
                "device": "email",
                "country": "US",
                "user_segment": row['user_segment'],
                "voucher_value": row['voucher_value']
            })
    
    return pd.concat([events_df, pd.DataFrame(voucher_events)], ignore_index=True)

def main():
    """Main function to generate synthetic data"""
    print("🚀 Generating synthetic customer personalization data...")
    
    # Generate base data
    products = generate_products(N_PRODUCTS)
    users = generate_user_profiles(N_USERS)
    
    print(f"Generated {len(products)} products and {len(users)} users")
    
    # Generate events
    all_events = []
    start_date = datetime.now() - timedelta(days=DAYS)
    
    for user in users:
        # Determine number of sessions based on user activity
        sessions_count = max(1, int(np.random.poisson(3 * user['activity_rate'])))
        
        for session in range(sessions_count):
            session_date = start_date + timedelta(
                days=random.randint(0, DAYS-1),
                hours=random.randint(0, 23)
            )
            
            session_events = generate_session_events(user, products, session_date)
            all_events.extend(session_events)
    
    # Convert to DataFrame
    events_df = pd.DataFrame(all_events)
    events_df = events_df.sort_values('timestamp').reset_index(drop=True)
    
    print(f"Generated {len(events_df)} events")
    
    # Add voucher campaigns
    events_df = simulate_voucher_campaigns(events_df)
    
    # Create sessions summary
    sessions_summary = events_df.groupby('session_id').agg({
        'user_id': 'first',
        'timestamp': ['min', 'max'],
        'event_type': 'count',
        'cart_value': 'max',
        'user_segment': 'first',
        'country': 'first',
        'device': 'first'
    }).reset_index()
    
    sessions_summary.columns = [
        'session_id', 'user_id', 'session_start', 'session_end', 
        'event_count', 'max_cart_value', 'user_segment', 'country', 'device'
    ]
    
    # Add session duration
    sessions_summary['session_duration_minutes'] = (
        sessions_summary['session_end'] - sessions_summary['session_start']
    ).dt.total_seconds() / 60
    
    # Determine if session had purchase
    purchase_sessions = events_df[events_df['event_type'] == 'purchase']['session_id'].unique()
    sessions_summary['had_purchase'] = sessions_summary['session_id'].isin(purchase_sessions)
    
    # Determine if session was abandoned (had add_to_cart but no purchase)
    cart_sessions = events_df[events_df['event_type'] == 'add_to_cart']['session_id'].unique()
    sessions_summary['had_cart'] = sessions_summary['session_id'].isin(cart_sessions)
    sessions_summary['abandoned_cart'] = (
        sessions_summary['had_cart'] & ~sessions_summary['had_purchase']
    )
    
    # Save data
    events_df.to_parquet('data/synthetic_events.parquet', index=False)
    sessions_summary.to_parquet('data/synthetic_sessions.parquet', index=False)
    
    # Save metadata
    metadata = {
        'generation_date': datetime.now().isoformat(),
        'n_users': len(users),
        'n_products': len(products),
        'n_events': len(events_df),
        'n_sessions': len(sessions_summary),
        'date_range_days': DAYS,
        'abandonment_rate': ABANDONMENT_RATE,
        'voucher_response_rate': VOUCHER_RESPONSE_RATE
    }
    
    with open('data/synthetic_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("✅ Data generation complete!")
    print(f"📊 Summary:")
    print(f"   - Events: {len(events_df):,}")
    print(f"   - Sessions: {len(sessions_summary):,}")
    print(f"   - Abandoned carts: {sessions_summary['abandoned_cart'].sum():,}")
    print(f"   - Purchase rate: {sessions_summary['had_purchase'].mean():.2%}")
    print(f"   - Abandonment rate: {sessions_summary['abandoned_cart'].mean():.2%}")
    
    return events_df, sessions_summary

if __name__ == "__main__":
    import os
    os.makedirs('data', exist_ok=True)
    events_df, sessions_df = main()
