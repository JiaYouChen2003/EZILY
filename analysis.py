import csv
import argparse

from cifar_100_label import get_superclass_label


def check_superclass_in_csv(file_path):
    try:
        count = 0
        same_superclass_count = 0
        with open(file_path, mode='r') as file:
            reader = csv.reader(file)
            next(reader)
            
            for row in reader:
                if len(row) < 2:
                    print(f"Skipping invalid row: {row}")
                    continue

                superclass_0 = get_superclass_label(row[0])
                superclass_1 = get_superclass_label(row[1])
                if superclass_0 == superclass_1:
                    same_superclass_count += 1
                count += 1
        print(f"Same superclass mistake {(100 * same_superclass_count / count):.4f}% ({same_superclass_count} out of {count})")

    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")


def main(suffix):
    file_path = f'mistake_predicted_result_{suffix}.csv'
    check_superclass_in_csv(file_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("suffix", help="Set to enable LLM usage")
    args = parser.parse_args()

    main(args.suffix)
