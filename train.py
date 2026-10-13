"""Compatibility launcher; the installable CLI lives in grl_yolo.cli."""
from grl_yolo.cli import export_checkpoints, main, make_detector, save_detector


if __name__ == "__main__":
    main()
