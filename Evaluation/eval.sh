python Adversarial_Play_Stage/vllm_generation_traindata.py \
    --question_path Adversarial_Play_Stage/100_goals_test.json \
    --save_path Evaluation/einstein_r1_raw_strategies_questions.json      \
    --model_path  r1_32b \
    --followup einstein \
    --bon bon

python Adversarial_Play_Stage/extract_json_strategies.py \
    --question_path Evaluation/einstein_r1_raw_strategies_questions.json  \
    --save_path Evaluation/einstein_r1_json_strategies_questions.json 

python Adversarial_Play_Stage/generate_target.py\
    --question_path Evaluation/einstein_r1_json_strategies_questions.json   \
    --save_path Evaluation/target_rr_responses.json    \
    --model_path RR

python Adversarial_Play_Stage/llama_guard.py \
    --question_path Evaluation/target_rr_responses.json    \
    --save_path Evaluation/safe_judge/     \
    --model_path Llama-Guard-3-8B \
    --group_rank 0 \
    --group_size 4 &

python Adversarial_Play_Stage/llama_guard.py \
    --question_path Evaluation/target_rr_responses.json    \
    --save_path Evaluation/safe_judge/     \
    --model_path Llama-Guard-3-8B \
    --group_rank 1 \
    --group_size 4 &

python Adversarial_Play_Stage/llama_guard.py \
    --question_path Evaluation/target_rr_responses.json    \
    --save_path Evaluation/safe_judge/     \
    --model_path Llama-Guard-3-8B \
    --group_rank 2 \
    --group_size 4 &

python Adversarial_Play_Stage/llama_guard.py \
    --question_path Evaluation/target_rr_responses.json    \
    --save_path Evaluation/safe_judge/     \
    --model_path Llama-Guard-3-8B \
    --group_rank 3 \
    --group_size 4 

python Adversarial_Play_Stage/sleep_one_minute.py

python Adversarial_Play_Stage/eval_rule_based.py \
    --question_path Evaluation/safe_judge/ \
    --save_path Evaluation/rule_based_eval_results.json      \
    --model_path Qwen2.5-72B-Instruct

python Evaluation/calculate_asr.py \
    --judge_path Evaluation/rule_based_eval_results.json    


