import json
import argparse


def test(question_path, save_path):
    with open(question_path,"r") as f:
        dataset = json.load(f)
    output_file = []
    for item in dataset:
        try:
            strategy = item['completions'].split('</strategy>')[0].replace('\n', '')
            method = item['completions'].split('<application step>')[1].split('</application step>')[0].replace('\n', '')
            example = item['completions'].split('<application example>')[1].replace('\n', '').replace('user:', '').replace('[User]', '').replace('**User**:', '')

            output_file.append({
        "theme task": item['task'],
        "Strategy of the PDF": strategy,
        "Method Definition": method,
        "Example": example,
        "Source": 'einstein_r1'
    })
        except:
            continue
    with open(save_path, 'w') as jsonfile:
        json.dump(output_file, jsonfile, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--question_path', type=str,
                        default="",
                        help='model name in the hub or local path')
    parser.add_argument('--save_path', type=str,
                        default="",
                        help='model name in the hub or local path')
    args = parser.parse_args()
    test(args.question_path, args.save_path)
