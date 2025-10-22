# Code_Unlearning_LLM
This repo is forked from [original repo](https://github.com/yaojin17/Unlearning_LLM/tree/main)

**Attention!: It seems that the program only accepts llama-based models, further tests needed**

## How to run
### Environment Setup
```
pip install -r requirements.txt
python setup.py install
```

### Prepare model
#### Option 1: Using models from hugging face

#### Option 2: Manually download model files
Either use Git Large File Storage or go to model's hugging face website and manually download all necessary files

### Prepare tokenized datasets
```
cd llm_unlearn/utils
python save_tokenized_dataset.py --tokenizer_name_or_path <name or path>
python ascent_plus_descent_tokenizer.py --tokenizer_name_or_path <name or path>
```

### Unlearning experiments
Export your wandb API key using `export WANDB_API_KEY=<your key>`
```
# Make sure you are under the llm_unlearn dir
torchrun --nproc_per_node=8 --master_port=20001  run_unlearn.py   \
    --target_model_name_or_path <name or path>  \
    --per_device_train_batch_size 1     \
    --do_unlearn  \
    --output_dir ./output \
    --overwrite_output_dir     \
    --num_train_epochs 1    \
    --logging_steps 1     \
    --learning_rate 2e-5     \
    --warmup_ratio 0.03 \
    --overwrite_cache \
    --save_total_limit 1 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --bf16 True \
    --tf32 True \
    --weight_decay 0. \
    --lr_scheduler_type "cosine" \
    --domain github \
    --gradient_accumulation_steps 85 \
    --unlearn_method gradient_ascent 
  ```
- Available domains with corresponding arguments: : 
  - `--domain arxiv  --gradient_accumulation_steps 60 `
  - `--domain github --gradient_accumulation_steps 85 `
- Available methods with corresponding arguments: 
  - `--unlearn_method gradient_ascent `
  - `--unlearn_method random_label --completely_random True` (named Fine-tuning with Random Labels in the paper)
  - `--unlearn_method random_label  --top_k 1  --rm_groundtruth True ` (named Unlearning with Adversarial Samples in the paper)
  - `--unlearn_method ascent_plus_descent`
  - `--unlearn_method ascent_plus_kl_divergence`
  - `--unlearn_method ascent_plus_descent --general True`
  - `--unlearn_method ascent_plus_kl_divergence --general True`

### Eval unlearned model
```
torchrun --nproc_per_node=8 --master_port=20001 run_eval.py \
    --model_name_or_path <path to unlearned model> \
    --per_device_eval_batch_size 1 \
    --do_eval \
    --output_dir ./output/github/Yi-6B-eval \
    --overwrite_output_dir \
    --overwrite_cache \
    --tf32 True \
    --domain github
```
### Membership inference attack
```
torchrun --nproc_per_node=8 --master_port=20001 run_mia.py \
        --model_name_or_path <path to unlearned model> \
        --per_device_eval_batch_size 1 \
        --do_eval \
        --output_dir ./output/arxiv/Yi-6B-mia \
        --overwrite_output_dir \
        --overwrite_cache \
        --tf32 True \
        --domain github
```

