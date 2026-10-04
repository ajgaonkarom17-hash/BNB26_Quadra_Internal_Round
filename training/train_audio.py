"""Train the AUDIO individual model head.

Usage:
    python -m training.train_audio --manifest data/manifests/audio.csv
"""
from training.individual import main

if __name__ == "__main__":
    main("audio")
