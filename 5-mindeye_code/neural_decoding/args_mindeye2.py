import argparse
import sys

def parse_args():
    parser = argparse.ArgumentParser(description="Model Training Configuration")

    ###### Frequently changing settings ######
    parser.add_argument(
        '--hidden_dim', type=int, default=1024,
        help="hidden_size"
    )

    # Jupyter 환경에서는 빈 리스트를 전달해야 실행이 됨
    if any("ipykernel_launcher" in arg for arg in sys.argv):
        args = parser.parse_args([])  
    else:
        args = parser.parse_args()

    return args