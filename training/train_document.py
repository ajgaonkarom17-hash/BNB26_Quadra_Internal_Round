"""Train the DOCUMENT individual model head.

Usage:
    python -m training.train_document --manifest data/manifests/document.csv
"""
from training.individual import main

if __name__ == "__main__":
    main("document")
