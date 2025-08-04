#!/usr/bin/env python3
"""
Temporary Public Sharing for MNIST App
Creates a temporary public URL using ngrok for immediate sharing
"""

from pyngrok import ngrok
import subprocess
import time
import threading

def run_streamlit():
    """Run Streamlit app"""
    subprocess.run(["streamlit", "run", "MNISTproj.py", "--server.port=8501"])

def create_public_url():
    """Create a temporary public URL"""
    print("🚀 Creating temporary public URL for your MNIST app...")
    
    # Start Streamlit in a separate thread
    streamlit_thread = threading.Thread(target=run_streamlit)
    streamlit_thread.daemon = True
    streamlit_thread.start()
    
    # Wait for Streamlit to start
    time.sleep(5)
    
    # Create ngrok tunnel
    public_url = ngrok.connect(8501)
    
    print("="*60)
    print("🎉 YOUR MNIST APP IS NOW PUBLICLY AVAILABLE!")
    print("="*60)
    print(f"🌐 Shareable URL: {public_url}")
    print("="*60)
    print("📋 Share this link with your friends!")
    print("⚠️  Note: This is a temporary URL that will expire when you close this script")
    print("💡 For permanent deployment, use Streamlit Community Cloud")
    print("="*60)
    
    try:
        # Keep the script running
        input("Press Enter to stop sharing...")
    except KeyboardInterrupt:
        pass
    finally:
        ngrok.kill()
        print("\n🛑 Sharing stopped. URL is no longer accessible.")

if __name__ == "__main__":
    create_public_url()
