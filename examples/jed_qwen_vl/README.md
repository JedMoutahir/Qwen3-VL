# Jed’s Qwen-VL Inference + Tiny Eval Table

Drop-in demo for **Qwen2.5-VL / Qwen3-VL** style models using 🤗 Transformers.
- Single/batch image inference
- Simple CSV-driven batch evaluation with a lexical F1 (token overlap) metric
- CPU works for smoke tests; GPU recommended for speed

## Quickstart

```bash
conda env create -f env.yml && conda activate qwen-vl
# Single image
python infer_qwen_vl.py   --model-id Qwen/Qwen2.5-VL-7B-Instruct   --images ./samples/doc1.png   --prompt "Extract the title and the main date."   --out runs/single

# Batch from CSV (image,question[,ref_answer])
python batch_eval.py   --model-id Qwen/Qwen2.5-VL-7B-Instruct   --csv examples/jed_qwen_vl/sample_prompts.csv   --out runs/batch
```
The batch script writes `preds.jsonl` and `eval.csv` (with per-row F1 and aggregate).

> Replace `--model-id` with any HF-compatible Qwen-VL checkpoint that supports `AutoProcessor` + `generate` with images.
