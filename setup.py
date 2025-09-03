"""
Setup script for AI Customer Personalization Platform
"""

import os
import subprocess
import sys
import json
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e.stderr}")
        return False

def create_directories():
    """Create necessary directories"""
    directories = ['data', 'models', 'logs']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"📁 Created directory: {directory}")

def install_dependencies():
    """Install Python dependencies"""
    return run_command("pip install -r requirements.txt", "Installing Python dependencies")

def setup_redis():
    """Setup Redis (instructions for different platforms)"""
    print("🔧 Redis Setup Instructions:")
    print("=" * 50)
    
    if sys.platform == "win32":
        print("Windows:")
        print("1. Download Redis from: https://github.com/microsoftarchive/redis/releases")
        print("2. Install and start Redis service")
        print("3. Or use Docker: docker run -d -p 6379:6379 redis:alpine")
    elif sys.platform == "darwin":
        print("macOS:")
        print("1. Install with Homebrew: brew install redis")
        print("2. Start Redis: brew services start redis")
        print("3. Or use Docker: docker run -d -p 6379:6379 redis:alpine")
    else:
        print("Linux:")
        print("1. Install Redis: sudo apt-get install redis-server")
        print("2. Start Redis: sudo systemctl start redis")
        print("3. Or use Docker: docker run -d -p 6379:6379 redis:alpine")
    
    print("\nTest Redis connection:")
    print("redis-cli ping")
    print("(Should return 'PONG')")

def generate_sample_data():
    """Generate sample data for testing"""
    print("📊 Generating sample data...")
    try:
        import pandas as pd
        import numpy as np
        from datetime import datetime, timedelta
        
        # Create sample events
        events = []
        users = [f"user_{i:03d}" for i in range(100)]
        products = [f"SKU_{i:03d}" for i in range(50)]
        
        for i in range(1000):
            events.append({
                'event_id': f"evt_{i:06d}",
                'user_id': np.random.choice(users),
                'session_id': f"sess_{i:06d}",
                'timestamp': datetime.now() - timedelta(days=np.random.randint(0, 30)),
                'event_type': np.random.choice(['page_view', 'product_view', 'add_to_cart', 'purchase']),
                'product_id': np.random.choice(products),
                'price': np.random.uniform(10, 200),
                'quantity': np.random.randint(1, 3),
                'cart_value': np.random.uniform(0, 500),
                'page_url': f"/product/{np.random.choice(products)}",
                'referrer': np.random.choice(['direct', 'google', 'facebook']),
                'device': np.random.choice(['desktop', 'mobile', 'tablet']),
                'country': np.random.choice(['US', 'UK', 'CA']),
                'user_segment': np.random.choice(['high_value', 'regular', 'occasional'])
            })
        
        events_df = pd.DataFrame(events)
        events_df.to_parquet('data/sample_events.parquet', index=False)
        
        # Create sample sessions
        sessions = events_df.groupby('session_id').agg({
            'user_id': 'first',
            'timestamp': ['min', 'max'],
            'event_type': 'count',
            'cart_value': 'max',
            'user_segment': 'first',
            'country': 'first',
            'device': 'first'
        }).reset_index()
        
        sessions.columns = [
            'session_id', 'user_id', 'session_start', 'session_end', 
            'event_count', 'max_cart_value', 'user_segment', 'country', 'device'
        ]
        
        sessions['session_duration_minutes'] = (
            sessions['session_end'] - sessions['session_start']
        ).dt.total_seconds() / 60
        
        sessions['had_purchase'] = np.random.choice([True, False], len(sessions), p=[0.2, 0.8])
        sessions['had_cart'] = np.random.choice([True, False], len(sessions), p=[0.4, 0.6])
        sessions['abandoned_cart'] = sessions['had_cart'] & ~sessions['had_purchase']
        
        sessions.to_parquet('data/sample_sessions.parquet', index=False)
        
        print("✅ Sample data generated successfully")
        return True
        
    except Exception as e:
        print(f"❌ Failed to generate sample data: {e}")
        return False

def create_docker_compose():
    """Create Docker Compose file for easy setup"""
    docker_compose_content = """version: '3.8'

services:
  redis:
    image: redis:alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    command: redis-server --appendonly yes

  api:
    build: .
    ports:
      - "8000:8000"
    depends_on:
      - redis
    environment:
      - REDIS_URL=redis://redis:6379/0
    volumes:
      - ./models:/app/models
      - ./data:/app/data
    command: python 3_realtime_decision_service.py

volumes:
  redis_data:
"""
    
    with open('docker-compose.yml', 'w') as f:
        f.write(docker_compose_content)
    
    print("✅ Created docker-compose.yml")

def create_dockerfile():
    """Create Dockerfile for containerization"""
    dockerfile_content = """FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["python", "3_realtime_decision_service.py"]
"""
    
    with open('Dockerfile', 'w') as f:
        f.write(dockerfile_content)
    
    print("✅ Created Dockerfile")

def create_startup_script():
    """Create startup script"""
    startup_script = """#!/bin/bash

echo "🚀 Starting AI Customer Personalization Platform..."

# Check if Redis is running
if ! redis-cli ping > /dev/null 2>&1; then
    echo "❌ Redis is not running. Please start Redis first."
    echo "   Docker: docker run -d -p 6379:6379 redis:alpine"
    echo "   Or install Redis locally"
    exit 1
fi

# Generate sample data if it doesn't exist
if [ ! -f "data/sample_events.parquet" ]; then
    echo "📊 Generating sample data..."
    python -c "
import sys
sys.path.append('.')
from setup import generate_sample_data
generate_sample_data()
"
fi

# Train models if they don't exist
if [ ! -f "models/conversion_model.joblib" ]; then
    echo "🤖 Training ML models..."
    python 2_training_pipeline.py
fi

# Start the API server
echo "🌐 Starting API server..."
python 3_realtime_decision_service.py
"""
    
    with open('start.sh', 'w') as f:
        f.write(startup_script)
    
    # Make executable on Unix systems
    if sys.platform != "win32":
        os.chmod('start.sh', 0o755)
    
    print("✅ Created start.sh")

def main():
    """Main setup function"""
    print("🚀 AI Customer Personalization Platform Setup")
    print("=" * 50)
    
    # Create directories
    create_directories()
    
    # Install dependencies
    if not install_dependencies():
        print("❌ Setup failed at dependency installation")
        return False
    
    # Setup Redis instructions
    setup_redis()
    
    # Generate sample data
    generate_sample_data()
    
    # Create Docker files
    create_docker_compose()
    create_dockerfile()
    
    # Create startup script
    create_startup_script()
    
    print("\n🎉 Setup completed successfully!")
    print("\n📋 Next Steps:")
    print("1. Start Redis (see instructions above)")
    print("2. Run: python 1_simulate_events.py (to generate training data)")
    print("3. Run: python 2_training_pipeline.py (to train ML models)")
    print("4. Run: python 3_realtime_decision_service.py (to start API)")
    print("5. Open demo_website.html in your browser")
    print("\n🐳 Or use Docker:")
    print("   docker-compose up --build")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
