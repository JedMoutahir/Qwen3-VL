import argparse, json, pandas as pd
from pathlib import Path
from typing import List, Dict
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM, AutoTokenizer
import numpy as np

def lexical_f1(pred: str, ref: str) -> float:
    def tokens(s): 
        return [t for t in ''.join(ch.lower() if ch.isalnum() else ' ' for ch in s).split() if t]
    p, r = set(tokens(pred)), set(tokens(ref))
    if not p and not r: return 1.0
    if not p or not r: return 0.0
    prec = len(p & r) / max(len(p), 1)
    rec  = len(p & r) / max(len(r), 1)
    if prec+rec == 0: return 0.0
    return 2*prec*rec/(prec+rec)

def pick_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--csv", required=True, help="CSV with columns: image,question[,ref]")
    ap.add_argument("--out", default="runs/qwen_vl_batch")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--dtype", type=str, default="bfloat16")
    args = ap.parse_args()

    out_dir = Path(args.out); (out_dir/"answers").mkdir(parents=True, exist_ok=True)
    device = pick_device()
    dtype = torch.bfloat16 if args.dtype.lower() in ("bfloat16","bf16") else (torch.float16 if args.dtype.lower() in ("float16","fp16") else torch.float32)
    print(f"Loading {args.model_id} on {device} dtype={dtype}")

    df = pd.read_csv(args.csv)
    if "image" not in df.columns or "question" not in df.columns:
        raise ValueError("CSV must have columns: image,question[,ref]")

    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, torch_dtype=dtype,
        device_map="auto" if device.startswith("cuda") else None,
        trust_remote_code=True,
    )
    if device == "cpu":
        model = model.to("cpu")

    preds = []
    for i, row in df.iterrows():
        img_path = Path(row["image"])
        if not img_path.exists():
            preds.append({"idx": int(i), "image": str(img_path), "question": row["question"], "answer": "", "ref": row.get("ref",""), "f1": np.nan})
            continue
        img = Image.open(img_path).convert("RGB")
        prompt = row["question"]
        if "<image>" not in prompt:
            prompt = "<image>\n" + prompt
        inputs = processor(text=[prompt], images=[img], return_tensors="pt")
        if device.startswith("cuda"):
            inputs = {k:(v.to(device) if hasattr(v,"to") else v) for k,v in inputs.items()}
        with torch.no_grad():
            out_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens, temperature=args.temperature)
        if hasattr(processor, "batch_decode"):
            text = processor.batch_decode(out_ids, skip_special_tokens=True)[0]
        else:
            tok = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
            text = tok.decode(out_ids[0], skip_special_tokens=True)

        rec = {"idx": int(i), "image": str(img_path), "question": row["question"], "answer": text, "ref": row.get("ref","")}
        if isinstance(row.get("ref", None), str) and row.get("ref","").strip():
            rec["f1"] = lexical_f1(text, row["ref"])
        else:
            rec["f1"] = np.nan
        preds.append(rec)
        (out_dir/"answers"/(img_path.stem + ".md")).write_text(f"# Q: {row['question']}\n\n**A:** {text}\n", encoding="utf-8")

    # write outputs
    with (out_dir/"preds.jsonl").open("w", encoding="utf-8") as f:
        for r in preds:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    pd.DataFrame(preds).to_csv(out_dir/"eval.csv", index=False)
    # aggregate
    agg = pd.DataFrame(preds)
    if "f1" in agg.columns:
        avg = float(agg["f1"].dropna().mean()) if agg["f1"].notna().any() else float("nan")
        print({"avg_f1": avg, "n": int(len(agg))})
    print(f"Wrote {out_dir/'eval.csv'}")

if __name__ == "__main__":
    main()
