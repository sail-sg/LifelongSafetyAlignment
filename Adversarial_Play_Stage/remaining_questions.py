import json
import random
import argparse

def test(goal_path, B_s_path, output_remaining_goal_path):

    with open(goal_path,"r") as f:
        all_questions = json.load(f)

    with open(B_s_path,"r") as f:
        used_questions = json.load(f)
        
    output_file = []
    save = set()
    
    # save 保存了 B_s 中成功的那些goal
    for item in used_questions:
        goal = item['instruction'].split('[an pure example jailbreak prompt on this new task:[')[1].split(']]')[0]
        if goal not in save:
            save.add(goal)

    # 将本轮需要达成的goal中，失败了的保留下来，以供下一轮继续
    for item in all_questions:
        if item['instruction'] not in save:
            output_file.append(item)

    with open(output_remaining_goal_path, 'w') as jsonfile:
        json.dump(output_file, jsonfile, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--goal_path', type=str,
                        default="",
                        help='model name in the hub or local path')
    parser.add_argument('--B_s_path', type=str,
                        default="",
                        help='model name in the hub or local path')
    parser.add_argument('--output_remaining_goal_path', type=str,
                        default="",
                        help='model name in the hub or local path')
    args = parser.parse_args()
    test(args.goal_path, args.B_s_path, args.output_remaining_goal_path)