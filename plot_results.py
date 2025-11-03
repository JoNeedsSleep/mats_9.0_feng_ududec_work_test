import argparse
from util import plot_truthfulqa_from_jsonl


def main():
    parser = argparse.ArgumentParser(description='Plot TruthfulQA comparison bar chart (from JSONL)')
    parser.add_argument('--jsonl_path', type=str, default='result/comparison_data.jsonl')
    parser.add_argument('--output_path', type=str, default='result/truthfulqa_plot.png')
    parser.add_argument('--title', type=str, default='TruthfulQA')
    parser.add_argument('--confidence', type=float, default=0.95, help='Confidence level for intervals, e.g., 0.95')
    args = parser.parse_args()

    plot_truthfulqa_from_jsonl(args.jsonl_path, args.output_path, args.title, args.confidence)


if __name__ == '__main__':
    main()


