import json
import random
import argparse

def test(remaining_goal_path, B_f_path, output_remaining_B_f):
    with open(remaining_goal_path,"r") as f:
        all_questions = json.load(f)

    with open(B_f_path,"r") as f:
        used_questions = json.load(f)
        

    output_file = []
    save = set()
    # remaining goals to achieve in next loop
    for item in all_questions:
        if item['instruction'] not in save:
            save.add(item['instruction'])

    #remaining B_f questions
    for item in used_questions:
        goal = item['instruction']
        if goal in save:
            output_file.append(item)

    with open(output_remaining_B_f, 'w') as jsonfile:
        json.dump(output_file, jsonfile, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--remaining_goal_path', type=str,
                        default="",
                        help='model name in the hub or local path')
    parser.add_argument('--B_f_path', type=str,
                        default="",
                        help='model name in the hub or local path')
    parser.add_argument('--output_remaining_B_f', type=str,
                        default="",
                        help='model name in the hub or local path')
    args = parser.parse_args()
    test(args.remaining_goal_path, args.B_f_path, args.output_remaining_B_f)