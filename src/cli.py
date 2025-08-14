import os, glob, json, time
from multiprocessing import Pool, cpu_count
from typing import Tuple
from tqdm import tqdm
from dataclasses import asdict
from .cropper import Settings, process_image_bytes

def is_image(p: str) -> bool:
    ext = os.path.splitext(p.lower())[1]
    return ext in [".jpg",".jpeg",".png",".bmp",".tif",".tiff",".webp"]

def _read_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()

def _process_one(args: Tuple[str, str, Settings]):
    path, out_dir, cfg = args
    name = os.path.splitext(os.path.basename(path))[0]
    out_path = os.path.join(out_dir, f"{name}.jpg")
    meta = {"file": path, "ok": False, "reason": "", "time_ms": None}
    t0 = time.time()
    try:
        data = _read_bytes(path)
        res = process_image_bytes(data, cfg)
        meta.update(res.get("meta", {}))
        meta["ok"] = bool(res.get("ok", False))
        meta["reason"] = res.get("reason", "")
        if res.get("ok") and "image_bytes" in res:
            os.makedirs(out_dir, exist_ok=True)
            with open(out_path, "wb") as f:
                f.write(res["image_bytes"])
    except Exception as e:
        meta["ok"] = False; meta["reason"] = f"exception:{e}"
    meta["time_ms"] = int(1000*(time.time()-t0))
    return meta

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Bulk Spanish ID cropper MVP")
    ap.add_argument("--inp", required=True, help="input folder (recursive)")
    ap.add_argument("--out", required=True, help="output folder")
    ap.add_argument("--workers", type=int, default=max(1, cpu_count()//2))
    ap.add_argument("--log", default="process_log.jsonl")
    ap.add_argument("--warp_w", type=int, default=1024)
    args = ap.parse_args()

    cfg = Settings(warp_w=args.warp_w)
    paths = [p for p in glob.glob(os.path.join(args.inp,"**","*"), recursive=True) if is_image(p)]
    if not paths:
        print("No images found."); return

    os.makedirs(args.out, exist_ok=True)
    with Pool(processes=args.workers) as pool, open(args.log,"w",encoding="utf-8") as f:
        for meta in tqdm(pool.imap_unordered(_process_one, [(p, args.out, cfg) for p in paths], chunksize=8), total=len(paths)):
            f.write(json.dumps(meta, ensure_ascii=False)+"\n")

if __name__ == "__main__":
    main()
