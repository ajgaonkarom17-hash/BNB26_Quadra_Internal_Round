"""Train the IMAGE individual model head. See training/individual.py for details.

Usage:
    python -m training.train_image --manifest data/manifests/image.csv
"""
from training.individual import main

if __name__ == "__main__":
    main("image")
