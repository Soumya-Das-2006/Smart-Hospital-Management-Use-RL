import os
import sys

# Ensure the root dir is evaluated so 'app' and 'core' resolve cleanly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from run import app as flask_app

def main():
    # Run loop wrapper to satisfy openenv validation checks structure
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", 7860))
    flask_app.run(host=host, port=port)

if __name__ == "__main__":
    main()
