import argparse, os, json
from pathlib import Path
from typing import List
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM, AutoTokenizer

def pick_device(user: str|None):
    if user:
        return user
    return "cuda" if torch.cuda.is_available() else "cpu"

def pick_dtype(user: str|None):
    s = (user or "bfloat16").lower()
    if s in ("bf16","bfloat16"): return torch.bfloat16
    if s in ("fp16","float16"): return torch.float16
    if s in ("fp32","float32"): return torch.float32
    return torch.bfloat16

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--images", nargs="+", required=True, help="image path(s) or globs")
    ap.add_argument("--prompt", type=str, default="Describe the content of the image briefly.")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--dtype", type=str, default="bfloat16")
    ap.add_argument("--out", type=str, default="runs/qwen_vl_single")
    args = ap.parse_args()

    out_dir = Path(args.out); (out_dir/"answers").mkdir(parents=True, exist_ok=True)
    device = pick_device(args.device)
    dtype = pick_dtype(args.dtype)

    print(f"Loading {args.model_id} on {device} dtype={dtype}")
    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, torch_dtype=dtype,
        device_map="auto" if device.startswith("cuda") else None,
        trust_remote_code=True,
    )
    if device == "cpu":
        model = model.to("cpu")

    # expand paths
    paths : List[Path] = []
    for pat in args.images:
        p = list(Path().glob(pat)) if any(ch in pat for ch in "*?[") else [Path(pat)]
        paths.extend([x for x in p if x.is_file()])
    if not paths:
        print("No images found."); return

    results = []
    for p in paths:
        image = Image.open(p).convert("RGB")
        prompt = args.prompt if "<image>" in args.prompt else f"<image>\n{args.prompt}"
        inputs = processor(text=[prompt], images=[image], return_tensors="pt")
        if device.startswith("cuda"):
            inputs = {k:(v.to(device) if hasattr(v,"to") else v) for k,v in inputs.items()}
        with torch.no_grad():
            out_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens, temperature=args.temperature)
        # decode
        if hasattr(processor, "batch_decode"):
            text = processor.batch_decode(out_ids, skip_special_tokens=True)[0]
        else:
            tok = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
            text = tok.decode(out_ids[0], skip_special_tokens=True)

        results.append({"image": str(p), "prompt": prompt, "answer": text})
        (out_dir/"answers"/(p.stem + ".md")).write_text(f"# Answer for {p.name}\n\n{text}\n", encoding="utf-8")
        print(f"{p.name}: {text[:160]}{'...' if len(text)>160 else ''}")

    with (out_dir/"preds.jsonl").open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {out_dir/'preds.jsonl'}")

if __name__ == "__main__":
    main()
