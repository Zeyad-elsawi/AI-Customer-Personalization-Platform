

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json

class SmartAIDecisionEngine:
    def __init__(self):
        # User segments with different conversion probabilities
        self.user_segments = {
            'high_value': {
                'base_conversion_rate': 0.4,
                'cart_value_threshold': 200,
                'max_discount': 0.15  # 15% max
            },
            'regular': {
                'base_conversion_rate': 0.2,
                'cart_value_threshold': 100,
                'max_discount': 0.20  # 20% max
            },
            'occasional': {
                'base_conversion_rate': 0.1,
                'cart_value_threshold': 50,
                'max_discount': 0.25  # 25% max
            }
        }
        
        # Discount tiers based on cart value
        self.discount_tiers = {
            'low': {'min_cart': 50, 'max_cart': 100, 'base_discount': 0.05},
            'medium': {'min_cart': 100, 'max_cart': 300, 'base_discount': 0.10},
            'high': {'min_cart': 300, 'max_cart': 1000, 'base_discount': 0.15},
            'premium': {'min_cart': 1000, 'max_cart': float('inf'), 'base_discount': 0.20}
        }
    
    def analyze_user_behavior(self, user_events):
        """Analyze user behavior to determine segment and conversion probability"""
        
        # Count different event types
        page_views = len([e for e in user_events if e.get('event_type') == 'page_view'])
        product_views = len([e for e in user_events if e.get('event_type') == 'product_view'])
        cart_adds = len([e for e in user_events if e.get('event_type') == 'add_to_cart'])
        
        # Calculate engagement score
        engagement_score = (page_views * 0.1 + product_views * 0.3 + cart_adds * 0.6)
        
        # Determine user segment based on behavior
        if engagement_score > 5 and cart_adds > 2:
            segment = 'high_value'
        elif engagement_score > 2 and cart_adds > 0:
            segment = 'regular'
        else:
            segment = 'occasional'
        
        return segment, engagement_score
    
    def calculate_conversion_probability(self, user_segment, cart_value, engagement_score, time_on_site):
        """Calculate the probability that user will convert without intervention"""
        
        base_rate = self.user_segments[user_segment]['base_conversion_rate']
        
        # Adjust based on cart value (higher cart = higher conversion chance)
        cart_multiplier = 1 + (cart_value / 1000) * 0.5
        
        # Adjust based on engagement
        engagement_multiplier = 1 + (engagement_score / 10) * 0.3
        
        # Adjust based on time on site (more time = more interest)
        time_multiplier = 1 + min(time_on_site / 300, 0.5)  # Max 50% boost for 5+ minutes
        
        # Calculate final probability
        conversion_prob = base_rate * cart_multiplier * engagement_multiplier * time_multiplier
        
        # Cap at 80% (no one is 100% certain to convert)
        return min(conversion_prob, 0.8)
    
    def calculate_expected_roi(self, cart_value, discount_percent, conversion_prob_without, conversion_prob_with):
        """Calculate expected ROI of sending a discount"""
        
        # Revenue without discount
        revenue_without = conversion_prob_without * cart_value
        
        # Revenue with discount (reduced by discount amount)
        discounted_price = cart_value * (1 - discount_percent)
        revenue_with = conversion_prob_with * discounted_price
        
        # Expected gain
        expected_gain = revenue_with - revenue_without
        
        return expected_gain
    
    def determine_optimal_discount(self, user_segment, cart_value, conversion_prob, engagement_score):
        """Determine the optimal discount amount and whether to send it"""
        
        segment_info = self.user_segments[user_segment]
        
        # Don't send discount if cart value is too low
        if cart_value < segment_info['cart_value_threshold']:
            return None, "Cart value too low for this user segment"
        
        # Don't send discount if conversion probability is already high
        if conversion_prob > 0.6:
            return None, "User likely to convert without discount"
        
        # Find appropriate discount tier
        discount_tier = None
        for tier_name, tier_info in self.discount_tiers.items():
            if tier_info['min_cart'] <= cart_value < tier_info['max_cart']:
                discount_tier = tier_info
                break
        
        if not discount_tier:
            return None, "Cart value too high for discount strategy"
        
        # Start with base discount for the tier
        base_discount = discount_tier['base_discount']
        
        # Adjust based on user segment (occasional users get higher discounts)
        if user_segment == 'occasional':
            base_discount *= 1.2
        elif user_segment == 'regular':
            base_discount *= 1.0
        else:  # high_value
            base_discount *= 0.8
        
        # Adjust based on conversion probability (lower probability = higher discount)
        if conversion_prob < 0.2:
            base_discount *= 1.3
        elif conversion_prob < 0.4:
            base_discount *= 1.1
        
        # Cap at segment maximum
        max_discount = segment_info['max_discount']
        final_discount = min(base_discount, max_discount)
        
        # Calculate expected ROI
        # Assume discount increases conversion probability by 20-40%
        conversion_boost = 0.2 + (final_discount * 2)  # Higher discount = higher boost
        new_conversion_prob = min(conversion_prob + conversion_boost, 0.9)
        
        expected_roi = self.calculate_expected_roi(
            cart_value, final_discount, conversion_prob, new_conversion_prob
        )
        
        # Only send discount if expected ROI is positive
        if expected_roi > 0:
            discount_percent = int(final_discount * 100)
            return discount_percent, f"Expected ROI: ${expected_roi:.2f}"
        else:
            return None, f"Negative ROI: ${expected_roi:.2f}"
    
    def make_decision(self, user_events, cart_value, time_on_site):
        """Main decision function - determines if and what discount to send"""
        
        # Analyze user behavior
        user_segment, engagement_score = self.analyze_user_behavior(user_events)
        
        # Calculate conversion probability
        conversion_prob = self.calculate_conversion_probability(
            user_segment, cart_value, engagement_score, time_on_site
        )
        
        # Determine optimal discount
        discount_percent, reason = self.determine_optimal_discount(
            user_segment, cart_value, conversion_prob, engagement_score
        )
        
        return {
            'should_send_discount': discount_percent is not None,
            'discount_percent': discount_percent,
            'user_segment': user_segment,
            'conversion_probability': conversion_prob,
            'engagement_score': engagement_score,
            'reason': reason,
            'cart_value': cart_value
        }

# Example usage and testing
if __name__ == "__main__":
    ai_engine = SmartAIDecisionEngine()
    
    # Test different scenarios
    test_cases = [
        {
            'name': 'High-value customer with expensive cart',
            'events': [
                {'event_type': 'page_view'}, {'event_type': 'page_view'},
                {'event_type': 'product_view'}, {'event_type': 'product_view'},
                {'event_type': 'add_to_cart'}, {'event_type': 'add_to_cart'},
                {'event_type': 'add_to_cart'}
            ],
            'cart_value': 800,
            'time_on_site': 180
        },
        {
            'name': 'Occasional customer with small cart',
            'events': [
                {'event_type': 'page_view'}, {'event_type': 'product_view'},
                {'event_type': 'add_to_cart'}
            ],
            'cart_value': 75,
            'time_on_site': 60
        },
        {
            'name': 'Regular customer with medium cart',
            'events': [
                {'event_type': 'page_view'}, {'event_type': 'page_view'},
                {'event_type': 'product_view'}, {'event_type': 'add_to_cart'},
                {'event_type': 'add_to_cart'}
            ],
            'cart_value': 150,
            'time_on_site': 120
        }
    ]
    
    print("🤖 Smart AI Decision Engine Test Results")
    print("=" * 50)
    
    for test_case in test_cases:
        result = ai_engine.make_decision(
            test_case['events'], 
            test_case['cart_value'], 
            test_case['time_on_site']
        )
        
        print(f"\n📊 {test_case['name']}")
        print(f"   Cart Value: ${test_case['cart_value']}")
        print(f"   User Segment: {result['user_segment']}")
        print(f"   Conversion Probability: {result['conversion_probability']:.2%}")
        print(f"   Engagement Score: {result['engagement_score']:.1f}")
        print(f"   Discount Decision: {result['discount_percent']}% off" if result['should_send_discount'] else "   Discount Decision: No discount")
        print(f"   Reason: {result['reason']}")
