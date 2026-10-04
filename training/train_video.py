"""Train the VIDEO individual model head.

Usage:
    python -m training.train_video --manifest data/manifests/video.csv
"""
from training.individual import main

if __name__ == "__main__":
    main("video")
