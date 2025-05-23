<div align="center">
<h2><a href="https://arxiv.org/abs/2504.15257">	
Lifelong Safety Alignment for Language Models</a></h2>

</a></h2>

Haoyu Wang, Zeyu Qin, Yifei Zhao, Chao Du, <br> Min Lin, Xueqian Wang, Tianyu Pang<sup>†</sup> 


<sup>†</sup> Corresponding Author
</div>

LLMs have made impressive progress, but their growing capabilities also expose them to highly flexible jailbreaking attacks designed to bypass safety alignment. While many existing defenses focus on known types of attacks, it is more critical to prepare LLMs for unseen attacks that may arise during deployment. To address this, we propose a lifelong safety alignment framework that enables LLMs to continuously adapt to new and evolving jailbreaking strategies. Our framework introduces a competitive setup between two components: a Meta-Attacker, trained to actively discover novel jailbreaking strategies, and a Defender, trained to resist them. To effectively warm up the Meta-Attacker, we first leverage the GPT-4o API to extract key insights from a large collection of jailbreak-related research papers. Through iterative training, the first iteration Meta-Attacker achieves a 73% attack success rate (ASR) on RR and a 57% transfer ASR on LAT using only single-turn attacks. Meanwhile, the Defender progressively improves its robustness and ultimately reduces the Meta-Attacker's success rate to just 7%, enabling safer and more reliable deployment of LLMs in open-ended environments.

<div align="center">
<p align="center">

<img src="image.png" width="600">

</p>
</div>


### 🚀 Installation

We follow the [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) to install the required dependencies, please run the following commands:

```bash
cd Train_Meta_Attacker_Defender/LLaMA-Factory
pip install -e ".[torch,metrics]" --no-build-isolation
cd ..
cd ..
pip install -r requirements.txt
```


### 🌴 Warm Up Stage

1. Download the jailbreak related papers
2. Set the correct path in strategy_pdfapi.sh
3. Set your openai api correctly
4. Bash strategy_pdfapi.sh

We provide an example on the past tense attack (2407.11969.pdf).

```
bash Warm_Up_Stage/strategy_pdfapi.sh
```
(We find currently Deepseek-R1 supports uploading files along with deep thinking mode. This powerful model achieves better summarization performance with our prompts when we compare it with GPT-4o API.)

### 🗡️ Adversarial-Play Stage

1. Download the Meta-Attacker and Defender ckpt, and set the correct path in PIPELINE.sh
2. For convenience, we provide the code to reproduce the result on R1-Qwen-32B and RR.
3. Manually set the beamsearch number.
4. Obtain the B_f and B_s to update the Meta-Attacker and Defender.

We provide an example on R1-Qwen-32B and RR, along with one beamsearch loop.
```
bash Adversarial_Play_Stage/PIPELINE.sh
```

### ❌ Refusal Generation
We give an example on generating refusals with LLaMA-3-8B-Instruct.
```
bash Refusal_Gen/refusal_gen.sh
```

### ♻️ Finetuning
We follow the LLaMA-Factory to do the finetuning. We provide the corresponding finetuning config in Train_Meta_Attacker_Defender/

### 🌹 Evaluation
To evaluate the unseen attack, use the test 100 goals. Set the correct Meta-Attacker and Defender path.

```
Bash Evaluation/eval.sh
```


## Acknowledgments

This repository is based on the codebase of [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory). We learn lessons from the prompts in [AutoDAN-Turbo](https://github.com/SaFoLab-WISC/AutoDAN-Turbo). Thanks for their impressive work!

