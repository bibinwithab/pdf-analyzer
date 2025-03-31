import os
import subprocess
import sys

def install_dependencies():
    """Installs dependencies from requirements.txt."""
    print("🔄 Installing dependencies...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

def create_venv():
    """Creates and activates a virtual environment."""
    print("🔄 Creating virtual environment...")
    os.system("python3 -m venv venv")

    activate_script = "venv/bin/activate" if os.name != "nt" else "venv\\Scripts\\activate"
    print(f"✅ Virtual environment created! Run 'source {activate_script}' to activate.")

if __name__ == "__main__":
    create_venv()
    install_dependencies()
