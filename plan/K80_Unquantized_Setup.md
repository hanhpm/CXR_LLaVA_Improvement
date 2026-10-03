# Official v2 checkpoint on the K80 server

Paper v3: https://arxiv.org/pdf/2310.18341v3 (Methods, pp. 5–7).
The paper uses ViT-L/16 and Llama-2 7B. The corresponding publicly released
checkpoint is ECOFRI/CXR-LLAVA-v2, pinned here to
`b2224786bb90d54b1e1291171866706cfbb44e2b`. This is not the deprecated
v1 RN50/13B model. Exact correspondence to the weights used for each paper table
is not independently established by this checkout.

Original checkpoint config, code, tokenizer, generation config and safetensors
are downloaded into `model_cache/cxr-llava-v2-original`. No bitsandbytes/NF4 is used.
The original files are preserved. `scripts/k80_original.py` makes a separate
compatibility copy with weight symlinks, replaces hardcoded BF16 casts with FP32,
keeps the per-layer KV cache on its assigned GPU, and changes streaming-thread
generation to synchronous generation so slow K80
tokens do not trigger the original 15-second streamer timeout.

Four visible GPUs are required. The 32 Llama layers are divided into groups of
eight; embeddings, image encoder, projector and output head use visible GPU 0.
This is sequential model dispatch, not data parallelism or tensor parallelism.
Original 512-pixel grayscale processing, normalization, report helper prompt,
512-token limit and helper defaults (`temperature=0.2`, `top_p=0.8`) are retained.
These defaults are from the released code, not verified paper evaluation settings.
Original tokenizer chat formatting is used by Transformers 4.36.2.

K80 does not support native BF16. FP32 and model dispatch are explicit hardware
adaptations, so results must be described as an unquantized K80 smoke test,
not exact paper reproduction or a clinical accuracy claim.

```bash
cd /storage/student4/hanhpm/CXR_LLaVA_Improvement
source /home/student4/anaconda3/etc/profile.d/conda.sh
# Only create if absent.
conda create -n cxr-llava-k80 python=3.10 pip -y
conda activate cxr-llava-k80
python -m pip install torch==1.12.1+cu102 --index-url https://download.pytorch.org/whl/cu102
python -m pip install -r requirements-k80.txt
python -m pip check
export CUDA_VISIBLE_DEVICES=0,1,2,3
python -c "import torch; print(torch.__version__, torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
python -m scripts.k80_original --download
python -u -m scripts.k80_original --execute --output result/k80_original_smoke_001
```

Use a new output path for each run. Output includes the actual prompt audit,
effective configuration, generated report and timing/VRAM. A one-image test
does not evaluate the paper's precision/recall/F1; that requires a reviewed cohort,
reference reports and official CheXpert labeling in a separate legacy environment.

## Verified on this server

One-image inference completed on four Tesla K80 GPUs with driver 440.33.01,
PyTorch 1.12.1+cu102 and Transformers 4.36.2. The successful output is
`result/k80_original_smoke_003/`. Generation took 27.59 seconds, excluding model
loading. Peak PyTorch allocated memory was 8.87 GiB on GPU 0 and 6.70 GiB on
each of GPUs 1–3; these are allocator measurements, not total nvidia-smi usage.
`pip check`, saved nonempty report, configuration and actual Llama-2 prompt
checks passed. Earlier directories 001/002 are incomplete attempts, not successes.
No pathology metrics, training or paper benchmark reproduction was performed.
