"""
F1 Race Strategy — Model-Free RL
Usage:
  python main.py train       # train all algorithms, save results
  python main.py visualize   # generate plots and GIF animations
  python main.py all         # train then visualize
"""

import sys


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"

    if mode in ("train", "all"):
        import train
        train.main()

    if mode in ("visualize", "all"):
        import visualize
        visualize.main()

    if mode not in ("train", "visualize", "all"):
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
