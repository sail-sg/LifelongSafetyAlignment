# This file contains the adversarial-play loop of the Meta-Attacker

python Adversarial_Play_Stage/vllm_generation_traindata.py \
    --question_path Adversarial_Play_Stage/4k_goals_train.json \
    --save_path Transfer_station/r1_32b/iteration1/einstein_r1_raw_strategies_questions.json      \
    --model_path  r1_32b \
    --followup einstein \
    --bon bon

python Adversarial_Play_Stage/extract_json_strategies.py \
    --question_path Transfer_station/r1_32b/iteration1/einstein_r1_raw_strategies_questions.json  \
    --save_path Transfer_station/r1_32b/iteration1/einstein_r1_json_strategies_questions.json 

python Adversarial_Play_Stage/generate_target.py\
    --question_path Transfer_station/r1_32b/iteration1/einstein_r1_json_strategies_questions.json   \
    --save_path Transfer_station/r1_32b/iteration1/target_rr_responses.json    \
    --model_path RR

python Adversarial_Play_Stage/llama_guard.py \
    --question_path Transfer_station/r1_32b/iteration1/target_rr_responses.json    \
    --save_path Transfer_station/r1_32b/iteration1/safe_judge/     \
    --model_path Llama-Guard-3-8B \
    --group_rank 0 \
    --group_size 4 &

python Adversarial_Play_Stage/llama_guard.py \
    --question_path Transfer_station/r1_32b/iteration1/target_rr_responses.json    \
    --save_path Transfer_station/r1_32b/iteration1/safe_judge/     \
    --model_path Llama-Guard-3-8B \
    --group_rank 1 \
    --group_size 4 &

python Adversarial_Play_Stage/llama_guard.py \
    --question_path Transfer_station/r1_32b/iteration1/target_rr_responses.json    \
    --save_path Transfer_station/r1_32b/iteration1/safe_judge/     \
    --model_path Llama-Guard-3-8B \
    --group_rank 2 \
    --group_size 4 &

python Adversarial_Play_Stage/llama_guard.py \
    --question_path Transfer_station/r1_32b/iteration1/target_rr_responses.json    \
    --save_path Transfer_station/r1_32b/iteration1/safe_judge/     \
    --model_path Llama-Guard-3-8B \
    --group_rank 3 \
    --group_size 4 

python Adversarial_Play_Stage/sleep_one_minute.py

python Adversarial_Play_Stage/eval_rule_based.py \
    --question_path Transfer_station/r1_32b/iteration1/safe_judge/ \
    --save_path Transfer_station/r1_32b/iteration1/rule_based_eval_results.json      \
    --model_path Qwen2.5-72B-Instruct

python Adversarial_Play_Stage/beam_search_r1.py \
    --raw_strategy_path Transfer_station/r1_32b/iteration1/einstein_r1_raw_strategies_questions.json \
    --eval_result_path Transfer_station/r1_32b/iteration1/rule_based_eval_results.json \
    --output_B_f_path Transfer_station/r1_32b/iteration1/B_f_0.json\
    --output_B_s_path  Transfer_station/r1_32b/iteration1/B_s_0.json\

python Adversarial_Play_Stage/remaining_questions.py \
    --goal_path Adversarial_Play_Stage/4k_goals_train.json \
    --B_s_path  Transfer_station/r1_32b/iteration1/B_s_0.json\
    --output_remaining_goal_path Transfer_station/r1_32b/iteration1/remaining_goals_beamsearch1.json

python Adversarial_Play_Stage/remaining_bon.py \
    --remaining_goal_path Transfer_station/r1_32b/iteration1/remaining_goals_beamsearch1.json\
    --B_f_path Transfer_station/r1_32b/iteration1/B_f_0.json\
    --output_remaining_B_f Transfer_station/r1_32b/iteration1/remaining_B_f_dialogs_beamsearch1.json






# repeat the following process 4 times, unless 47% goals are achieved
python Adversarial_Play_Stage/vllm_generation_traindata.py  \
    --question_path  Transfer_station/r1_32b/iteration1/remaining_B_f_dialogs_beamsearch1.json \
    --save_path Transfer_station/r1_32b/iteration1/einstein_r1_raw_strategies_questions_beamsearch1.json      \
    --model_path  r1_32b \
    --followup beamsearch

python Adversarial_Play_Stage/extract_json_strategies_beamsearch.py \
    --question_path Transfer_station/r1_32b/iteration1/einstein_r1_raw_strategies_questions_beamsearch1.json  \
    --save_path Transfer_station/r1_32b/iteration1/einstein_r1_json_strategies_questions_beamsearch1.json 

python Adversarial_Play_Stage/generate_target.py\
    --question_path Transfer_station/r1_32b/iteration1/einstein_r1_json_strategies_questions_beamsearch1.json   \
    --save_path Transfer_station/r1_32b/iteration1/target_rr_responses_beamsearch1.json    \
    --model_path RR

python Adversarial_Play_Stage/llama_guard.py \
    --question_path Transfer_station/r1_32b/iteration1/target_rr_responses_beamsearch1.json    \
    --save_path Transfer_station/r1_32b/iteration1/safe_judge_beam_search1/     \
    --model_path Llama-Guard-3-8B \
    --group_rank 0 \
    --group_size 4 &

python Adversarial_Play_Stage/llama_guard.py \
    --question_path Transfer_station/r1_32b/iteration1/target_rr_responses_beamsearch1.json    \
    --save_path Transfer_station/r1_32b/iteration1/safe_judge_beam_search1/     \
    --model_path Llama-Guard-3-8B \
    --group_rank 1 \
    --group_size 4 &

python Adversarial_Play_Stage/llama_guard.py \
    --question_path Transfer_station/r1_32b/iteration1/target_rr_responses_beamsearch1.json    \
    --save_path Transfer_station/r1_32b/iteration1/safe_judge_beam_search1/     \
    --model_path Llama-Guard-3-8B \
    --group_rank 2 \
    --group_size 4 &

python Adversarial_Play_Stage/llama_guard.py \
    --question_path Transfer_station/r1_32b/iteration1/target_rr_responses_beamsearch1.json    \
    --save_path Transfer_station/r1_32b/iteration1/safe_judge_beam_search1/     \
    --model_path Llama-Guard-3-8B \
    --group_rank 3 \
    --group_size 4 

python Adversarial_Play_Stage/sleep_one_minute.py

python Adversarial_Play_Stage/eval_rule_based.py \
    --question_path Transfer_station/r1_32b/iteration1/safe_judge_beam_search1/ \
    --save_path Transfer_station/r1_32b/iteration1/rule_based_eval_results_beamsearch1.json      \
    --model_path Qwen2.5-72B-Instruct

python Adversarial_Play_Stage/beam_search_r1_1.py \
    --raw_strategy_path Transfer_station/r1_32b/iteration1/einstein_r1_raw_strategies_questions_beamsearch1.json \
    --eval_result_path Transfer_station/r1_32b/iteration1/rule_based_eval_results_beamsearch1.json \
    --output_B_f_path Transfer_station/r1_32b/iteration1/B_f_1.json\
    --output_B_s_path  Transfer_station/r1_32b/iteration1/B_s_1.json\

python Adversarial_Play_Stage/remaining_questions.py \
    --goal_path Adversarial_Play_Stage/4k_goals_train.json \
    --B_s_path  Transfer_station/r1_32b/iteration1/B_s_1.json\
    --output_remaining_goal_path Transfer_station/r1_32b/iteration1/remaining_goals_beamsearch2.json

python Adversarial_Play_Stage/remaining_bon.py \
    --remaining_goal_path Transfer_station/r1_32b/iteration1/remaining_goals_beamsearch2.json\
    --B_f_path Transfer_station/r1_32b/iteration1/B_f_1.json\
    --output_remaining_B_f Transfer_station/r1_32b/iteration1/remaining_B_f_dialogs_beamsearch2.json