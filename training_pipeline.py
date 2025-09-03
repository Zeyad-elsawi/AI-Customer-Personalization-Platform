

import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    precision_recall_curve, roc_curve, accuracy_score
)
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, Any
import warnings
warnings.filterwarnings('ignore')

class CustomerPersonalizationTrainer:
    def __init__(self):
        self.conversion_model = None
        self.voucher_response_model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_importance = {}
        
    def load_data(self, events_path: str, sessions_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load synthetic data"""
        print("📊 Loading synthetic data...")
        events_df = pd.read_parquet(events_path)
        sessions_df = pd.read_parquet(sessions_path)
        
        print(f"Loaded {len(events_df):,} events and {len(sessions_df):,} sessions")
        return events_df, sessions_df
    
    def engineer_features(self, events_df: pd.DataFrame, sessions_df: pd.DataFrame) -> pd.DataFrame:
        """Engineer features for ML models"""
        print("🔧 Engineering features...")
        
        # Session-level features
        session_features = sessions_df.copy()
        
        # User-level historical features (RFM analysis)
        user_features = self._compute_user_rfm_features(events_df)
        session_features = session_features.merge(user_features, on='user_id', how='left')
        
        # Behavioral features
        behavioral_features = self._compute_behavioral_features(events_df, sessions_df)
        session_features = session_features.merge(behavioral_features, on='session_id', how='left')
        
        # Temporal features
        session_features = self._add_temporal_features(session_features)
        
        # Product interaction features
        product_features = self._compute_product_interaction_features(events_df, sessions_df)
        session_features = session_features.merge(product_features, on='session_id', how='left')
        
        # Fill missing values
        numeric_columns = session_features.select_dtypes(include=[np.number]).columns
        session_features[numeric_columns] = session_features[numeric_columns].fillna(0)
        
        print(f"Feature engineering complete. Dataset shape: {session_features.shape}")
        return session_features
    
    def _compute_user_rfm_features(self, events_df: pd.DataFrame) -> pd.DataFrame:
        """Compute Recency, Frequency, Monetary features for users"""
        now = events_df['timestamp'].max()
        
        # Get purchase events
        purchases = events_df[events_df['event_type'] == 'purchase'].copy()
        
        if len(purchases) == 0:
            # Create dummy RFM features if no purchases
            users = events_df['user_id'].unique()
            return pd.DataFrame({
                'user_id': users,
                'recency_days': 999,
                'frequency': 0,
                'monetary_value': 0,
                'avg_order_value': 0,
                'days_since_first_purchase': 999,
                'total_orders': 0
            })
        
        # Group by user and compute RFM
        user_rfm = purchases.groupby('user_id').agg({
            'timestamp': ['max', 'min', 'count'],
            'price': ['sum', 'mean']
        }).reset_index()
        
        user_rfm.columns = [
            'user_id', 'last_purchase_date', 'first_purchase_date', 
            'frequency', 'monetary_value', 'avg_order_value'
        ]
        
        # Compute recency
        user_rfm['recency_days'] = (now - user_rfm['last_purchase_date']).dt.days
        user_rfm['days_since_first_purchase'] = (now - user_rfm['first_purchase_date']).dt.days
        user_rfm['total_orders'] = user_rfm['frequency']
        
        # Add users with no purchases
        all_users = events_df['user_id'].unique()
        users_with_purchases = user_rfm['user_id'].unique()
        users_without_purchases = set(all_users) - set(users_with_purchases)
        
        if users_without_purchases:
            no_purchase_df = pd.DataFrame({
                'user_id': list(users_without_purchases),
                'last_purchase_date': None,
                'first_purchase_date': None,
                'frequency': 0,
                'monetary_value': 0,
                'avg_order_value': 0,
                'recency_days': 999,
                'days_since_first_purchase': 999,
                'total_orders': 0
            })
            user_rfm = pd.concat([user_rfm, no_purchase_df], ignore_index=True)
        
        return user_rfm[['user_id', 'recency_days', 'frequency', 'monetary_value', 
                        'avg_order_value', 'days_since_first_purchase', 'total_orders']]
    
    def _compute_behavioral_features(self, events_df: pd.DataFrame, sessions_df: pd.DataFrame) -> pd.DataFrame:
        """Compute behavioral features for sessions"""
        behavioral_features = []
        
        for session_id in sessions_df['session_id']:
            session_events = events_df[events_df['session_id'] == session_id]
            
            features = {
                'session_id': session_id,
                'page_views': len(session_events[session_events['event_type'] == 'page_view']),
                'product_views': len(session_events[session_events['event_type'] == 'product_view']),
                'add_to_cart_events': len(session_events[session_events['event_type'] == 'add_to_cart']),
                'remove_from_cart_events': len(session_events[session_events['event_type'] == 'remove_from_cart']),
                'checkout_attempts': len(session_events[session_events['event_type'] == 'checkout_started']),
                'unique_products_viewed': session_events[session_events['event_type'] == 'product_view']['product_id'].nunique(),
                'unique_categories_viewed': session_events[session_events['event_type'] == 'product_view']['product_category'].nunique(),
                'avg_product_price_viewed': session_events[session_events['event_type'] == 'product_view']['price'].mean() if len(session_events[session_events['event_type'] == 'product_view']) > 0 else 0,
                'max_product_price_viewed': session_events[session_events['event_type'] == 'product_view']['price'].max() if len(session_events[session_events['event_type'] == 'product_view']) > 0 else 0,
                'bounce_rate': 1 if len(session_events) == 1 else 0,  # Single event = bounce
                'cart_abandonment_events': len(session_events[session_events['event_type'] == 'remove_from_cart'])
            }
            
            behavioral_features.append(features)
        
        return pd.DataFrame(behavioral_features)
    
    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add temporal features"""
        df = df.copy()
        df['hour_of_day'] = df['session_start'].dt.hour
        df['day_of_week'] = df['session_start'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['is_business_hours'] = df['hour_of_day'].between(9, 17).astype(int)
        df['month'] = df['session_start'].dt.month
        df['quarter'] = df['session_start'].dt.quarter
        
        return df
    
    def _compute_product_interaction_features(self, events_df: pd.DataFrame, sessions_df: pd.DataFrame) -> pd.DataFrame:
        """Compute product interaction features"""
        product_features = []
        
        for session_id in sessions_df['session_id']:
            session_events = events_df[events_df['session_id'] == session_id]
            cart_events = session_events[session_events['event_type'] == 'add_to_cart']
            
            features = {
                'session_id': session_id,
                'cart_items_count': len(cart_events),
                'cart_value': cart_events['cart_value'].max() if len(cart_events) > 0 else 0,
                'avg_cart_item_price': cart_events['price'].mean() if len(cart_events) > 0 else 0,
                'cart_price_std': cart_events['price'].std() if len(cart_events) > 0 else 0,
                'cart_categories_count': cart_events['product_category'].nunique() if len(cart_events) > 0 else 0,
                'cart_brands_count': cart_events['product_brand'].nunique() if len(cart_events) > 0 else 0
            }
            
            product_features.append(features)
        
        return pd.DataFrame(product_features)
    
    def prepare_conversion_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for conversion prediction model"""
        print("🎯 Preparing conversion prediction data...")
        
        # Filter sessions with cart activity
        cart_sessions = df[df['had_cart'] == True].copy()
        
        # Create labels: convert within 7 days after session
        cart_sessions['converted_within_7d'] = 0
        
        # For simplicity, we'll use a heuristic based on user segment and cart value
        # In real implementation, you'd track actual conversions
        conversion_probability = (
            (cart_sessions['user_segment'] == 'high_value') * 0.3 +
            (cart_sessions['user_segment'] == 'regular') * 0.15 +
            (cart_sessions['user_segment'] == 'occasional') * 0.05 +
            (cart_sessions['max_cart_value'] > 100) * 0.2 +
            (cart_sessions['max_cart_value'] > 200) * 0.1
        )
        
        cart_sessions['converted_within_7d'] = np.random.binomial(1, conversion_probability)
        
        # Select features for conversion model
        feature_columns = [
            'max_cart_value', 'event_count', 'session_duration_minutes',
            'recency_days', 'frequency', 'monetary_value', 'avg_order_value',
            'page_views', 'product_views', 'add_to_cart_events', 'unique_products_viewed',
            'avg_product_price_viewed', 'bounce_rate',
            'hour_of_day', 'day_of_week', 'is_weekend', 'is_business_hours'
        ]
        
        # Handle categorical variables
        categorical_columns = ['user_segment', 'country', 'device']
        for col in categorical_columns:
            if col in cart_sessions.columns:
                le = LabelEncoder()
                cart_sessions[f'{col}_encoded'] = le.fit_transform(cart_sessions[col].astype(str))
                self.label_encoders[f'{col}_conversion'] = le
                feature_columns.append(f'{col}_encoded')
        
        X = cart_sessions[feature_columns].fillna(0)
        y = cart_sessions['converted_within_7d']
        
        print(f"Conversion dataset: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"Conversion rate: {y.mean():.2%}")
        
        return X.values, y.values
    
    def prepare_voucher_response_data(self, events_df: pd.DataFrame, sessions_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for voucher response model"""
        print("🎫 Preparing voucher response data...")
        
        # Find sessions that received vouchers
        voucher_events = events_df[events_df['event_type'] == 'voucher_sent']
        
        if len(voucher_events) == 0:
            print("No voucher events found, creating synthetic data...")
            # Create synthetic voucher response data
            cart_sessions = sessions_df[sessions_df['had_cart'] == True].sample(100)
            voucher_sessions = cart_sessions.copy()
            voucher_sessions['voucher_value'] = np.random.uniform(5, 50, len(voucher_sessions))
            voucher_sessions['voucher_redeemed'] = np.random.binomial(1, 0.15, len(voucher_sessions))
        else:
            voucher_sessions = voucher_events.merge(
                sessions_df, on=['user_id', 'session_id'], how='left'
            )
        
        # Select features for voucher response model
        feature_columns = [
            'max_cart_value', 'event_count', 'session_duration_minutes',
            'recency_days', 'frequency', 'monetary_value', 'avg_order_value',
            'page_views', 'product_views', 'add_to_cart_events', 'unique_products_viewed',
            'avg_product_price_viewed', 'bounce_rate',
            'hour_of_day', 'day_of_week', 'is_weekend', 'is_business_hours'
        ]
        
        # Add voucher-specific features
        if 'voucher_value' in voucher_sessions.columns:
            feature_columns.append('voucher_value')
            voucher_sessions['voucher_value'] = voucher_sessions['voucher_value'].fillna(0)
        else:
            voucher_sessions['voucher_value'] = 0
        
        # Handle categorical variables
        categorical_columns = ['user_segment', 'country', 'device']
        for col in categorical_columns:
            if col in voucher_sessions.columns:
                le = LabelEncoder()
                voucher_sessions[f'{col}_encoded'] = le.fit_transform(voucher_sessions[col].astype(str))
                self.label_encoders[f'{col}_voucher'] = le
                feature_columns.append(f'{col}_encoded')
        
        X = voucher_sessions[feature_columns].fillna(0)
        y = voucher_sessions.get('voucher_redeemed', np.random.binomial(1, 0.15, len(voucher_sessions)))
        
        print(f"Voucher response dataset: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"Voucher redemption rate: {y.mean():.2%}")
        
        return X, y
    
    def train_conversion_model(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Train conversion prediction model"""
        print("🤖 Training conversion prediction model...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train LightGBM model
        train_data = lgb.Dataset(X_train_scaled, label=y_train)
        val_data = lgb.Dataset(X_test_scaled, label=y_test, reference=train_data)
        
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': 42
        }
        
        self.conversion_model = lgb.train(
            params,
            train_data,
            valid_sets=[val_data],
            num_boost_round=1000,
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
        )
        
        # Evaluate model
        y_pred_proba = self.conversion_model.predict(X_test_scaled)
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        auc_score = roc_auc_score(y_test, y_pred_proba)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Conversion Model Performance:")
        print(f"  AUC: {auc_score:.3f}")
        print(f"  Accuracy: {accuracy:.3f}")
        print(f"  Classification Report:")
        print(classification_report(y_test, y_pred))
        
        # Feature importance
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': self.conversion_model.feature_importance()
        }).sort_values('importance', ascending=False)
        
        self.feature_importance['conversion'] = importance_df
        
        return {
            'auc': auc_score,
            'accuracy': accuracy,
            'feature_importance': importance_df.head(10).to_dict('records')
        }
    
    def train_voucher_response_model(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Train voucher response model"""
        print("🎫 Training voucher response model...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train Random Forest model
        self.voucher_response_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight='balanced'
        )
        
        self.voucher_response_model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        y_pred_proba = self.voucher_response_model.predict_proba(X_test_scaled)[:, 1]
        y_pred = self.voucher_response_model.predict(X_test_scaled)
        
        auc_score = roc_auc_score(y_test, y_pred_proba)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Voucher Response Model Performance:")
        print(f"  AUC: {auc_score:.3f}")
        print(f"  Accuracy: {accuracy:.3f}")
        print(f"  Classification Report:")
        print(classification_report(y_test, y_pred))
        
        # Feature importance
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': self.voucher_response_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        self.feature_importance['voucher_response'] = importance_df
        
        return {
            'auc': auc_score,
            'accuracy': accuracy,
            'feature_importance': importance_df.head(10).to_dict('records')
        }
    
    def save_models(self, output_dir: str = 'models'):
        """Save trained models and metadata"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Save models
        if self.conversion_model:
            joblib.dump(self.conversion_model, f'{output_dir}/conversion_model.joblib')
            print(f"✅ Saved conversion model to {output_dir}/conversion_model.joblib")
        
        if self.voucher_response_model:
            joblib.dump(self.voucher_response_model, f'{output_dir}/voucher_response_model.joblib')
            print(f"✅ Saved voucher response model to {output_dir}/voucher_response_model.joblib")
        
        # Save scaler and encoders
        joblib.dump(self.scaler, f'{output_dir}/scaler.joblib')
        joblib.dump(self.label_encoders, f'{output_dir}/label_encoders.joblib')
        
        # Save feature importance
        with open(f'{output_dir}/feature_importance.json', 'w') as f:
            json.dump({
                'conversion': self.feature_importance.get('conversion', {}).to_dict('records') if 'conversion' in self.feature_importance else [],
                'voucher_response': self.feature_importance.get('voucher_response', {}).to_dict('records') if 'voucher_response' in self.feature_importance else []
            }, f, indent=2)
        
        print(f"✅ Saved all models and metadata to {output_dir}/")
    
    def plot_feature_importance(self, model_type: str = 'conversion'):
        """Plot feature importance"""
        if model_type not in self.feature_importance:
            print(f"No feature importance data for {model_type}")
            return
        
        importance_df = self.feature_importance[model_type].head(15)
        
        plt.figure(figsize=(10, 8))
        sns.barplot(data=importance_df, x='importance', y='feature')
        plt.title(f'Feature Importance - {model_type.title()} Model')
        plt.xlabel('Importance')
        plt.tight_layout()
        plt.show()

def main():
    """Main training pipeline"""
    print("🚀 Starting AI Customer Personalization Training Pipeline")
    
    # Initialize trainer
    trainer = CustomerPersonalizationTrainer()
    
    # Load data
    events_df, sessions_df = trainer.load_data(
        'data/synthetic_events.parquet',
        'data/synthetic_sessions.parquet'
    )
    
    # Engineer features
    features_df = trainer.engineer_features(events_df, sessions_df)
    
    # Prepare conversion data
    X_conv, y_conv = trainer.prepare_conversion_data(features_df)
    
    # Train conversion model
    conv_results = trainer.train_conversion_model(X_conv, y_conv)
    
    # Prepare voucher response data
    X_vouch, y_vouch = trainer.prepare_voucher_response_data(events_df, features_df)
    
    # Train voucher response model
    vouch_results = trainer.train_voucher_response_model(X_vouch, y_vouch)
    
    # Save models
    trainer.save_models()
    
    # Plot feature importance
    trainer.plot_feature_importance('conversion')
    trainer.plot_feature_importance('voucher_response')
    
    print("🎉 Training pipeline complete!")
    print(f"📊 Results Summary:")
    print(f"   Conversion Model AUC: {conv_results['auc']:.3f}")
    print(f"   Voucher Response Model AUC: {vouch_results['auc']:.3f}")
    
    return trainer

if __name__ == "__main__":
    trainer = main()
