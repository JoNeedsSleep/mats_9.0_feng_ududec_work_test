import argparse
from util import plot_truthfulqa_from_json


def main():
    parser = argparse.ArgumentParser(description='Plot TruthfulQA comparison bar chart')
    parser.add_argument('--json_path', type=str, default='result/comparison_data.json')
    parser.add_argument('--output_path', type=str, default='result/truthfulqa_plot.png')
    parser.add_argument('--title', type=str, default='TruthfulQA')
    args = parser.parse_args()

    plot_truthfulqa_from_json(args.json_path, args.output_path, args.title)


if __name__ == '__main__':
    main()


