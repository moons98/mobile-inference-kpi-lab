#!/usr/bin/env python3
"""
Image quality evaluation: LPIPS + CLIP Score.

Base image 기준으로 비교 이미지들의 LPIPS와 CLIP Score를 계산한다.
LPIPS: base image 대비 지각적 유사도 (낮을수록 유사)
CLIP Score: 각 이미지의 프롬프트 부합도 (높을수록 좋음)

Usage:
    # LPIPS only (base image 대비 비교)
    python scripts/eval/image_quality.py \\
        --base results/sd_50step_fp32.png \\
        --images results/lcm_4step.png results/lcm_8step.png

    # CLIP Score only (프롬프트 필요)
    python scripts/eval/image_quality.py \\
        --images results/lcm_4step.png results/lcm_8step.png \\
        --prompt "a cat sitting on a red chair"

    # LPIPS + CLIP Score 둘 다
    python scripts/eval/image_quality.py \\
        --base results/sd_50step_fp32.png \\
        --images results/lcm_4step.png results/lcm_8step.png \\
        --prompt "a cat sitting on a red chair"

    # 디렉토리 내 이미지 전체 평가
    python scripts/eval/image_quality.py \\
        --base results/sd_50step_fp32.png \\
        --images-dir results/ \\
        --prompt "a cat sitting on a red chair"

    # CSV 저장
    python scripts/eval/image_quality.py \\
        --base results/sd_50step_fp32.png \\
        --images results/lcm_4step.png results/lcm_8step.png \\
        --prompt "a cat sitting on a red chair" \\
        --output results/quality_eval.csv
"""

import argparse
import csv
import sys
from pathlib import Path

import torch
from PIL import Image


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def load_image(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def to_tensor_lpips(img: Image.Image) -> torch.Tensor:
    """LPIPS 입력: [-1, 1] 범위, (1, 3, H, W)"""
    import torchvision.transforms.functional as TF
    t = TF.to_tensor(img)  # [0, 1]
    t = t * 2.0 - 1.0      # [-1, 1]
    return t.unsqueeze(0)


def to_tensor_clip(img: Image.Image, size: int = 224) -> torch.Tensor:
    """CLIP 입력: 224×224, ImageNet 정규화, (1, 3, 224, 224), uint8 형식으로 torchmetrics에 전달"""
    import torchvision.transforms.functional as TF
    t = TF.to_tensor(img.resize((size, size), Image.BICUBIC))
    return (t * 255).to(torch.uint8).unsqueeze(0)


def compute_lpips(base_img: Image.Image, compare_imgs: list[Image.Image], device: torch.device) -> list[float]:
    import lpips
    loss_fn = lpips.LPIPS(net="alex").to(device)
    loss_fn.eval()

    base_t = to_tensor_lpips(base_img).to(device)
    scores = []
    with torch.no_grad():
        for img in compare_imgs:
            # LPIPS는 동일 해상도 필요 — base 크기에 맞춤
            if img.size != base_img.size:
                img = img.resize(base_img.size, Image.BICUBIC)
            t = to_tensor_lpips(img).to(device)
            score = loss_fn(base_t, t).item()
            scores.append(score)
    return scores


def compute_clip_scores(imgs: list[Image.Image], prompt: str, device: torch.device) -> list[float]:
    from torchmetrics.functional.multimodal import clip_score
    from functools import partial

    clip_score_fn = partial(
        clip_score,
        model_name_or_path="openai/clip-vit-base-patch16",
    )

    scores = []
    for img in imgs:
        t = to_tensor_clip(img).to(device)
        score = clip_score_fn(t, [prompt]).item()
        scores.append(round(score, 4))
    return scores


def collect_images(images: list[str] | None, images_dir: str | None) -> list[Path]:
    paths = []
    if images:
        for p in images:
            path = Path(p)
            if not path.exists():
                print(f"[경고] 파일 없음: {path}", file=sys.stderr)
            else:
                paths.append(path)
    if images_dir:
        dir_path = Path(images_dir)
        if not dir_path.is_dir():
            print(f"[오류] 디렉토리 없음: {dir_path}", file=sys.stderr)
        else:
            paths.extend(
                sorted(p for p in dir_path.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
            )
    return paths


def print_results(rows: list[dict]) -> None:
    has_lpips = any(r["lpips"] is not None for r in rows)
    has_clip = any(r["clip_score"] is not None for r in rows)

    headers = ["image"]
    if has_lpips:
        headers.append("lpips ↓")
    if has_clip:
        headers.append("clip_score ↑")

    col_widths = [max(len(h), max(len(r["image"]) for r in rows)) for h in headers]
    col_widths[0] = max(col_widths[0], max(len(r["image"]) for r in rows))

    def fmt_row(vals):
        return "  ".join(str(v).ljust(w) for v, w in zip(vals, col_widths))

    print()
    print(fmt_row(headers))
    print("  ".join("-" * w for w in col_widths))
    for r in rows:
        vals = [r["image"]]
        if has_lpips:
            vals.append(f"{r['lpips']:.4f}" if r["lpips"] is not None else "—")
        if has_clip:
            vals.append(f"{r['clip_score']:.4f}" if r["clip_score"] is not None else "—")
        print(fmt_row(vals))
    print()


def save_csv(rows: list[dict], output: str) -> None:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "lpips", "clip_score"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV 저장: {path}")


def main():
    parser = argparse.ArgumentParser(description="LPIPS + CLIP Score 평가")
    parser.add_argument("--base", type=str, help="LPIPS 기준 이미지 경로")
    parser.add_argument("--images", nargs="+", type=str, help="평가할 이미지 경로 목록")
    parser.add_argument("--images-dir", type=str, help="평가할 이미지가 있는 디렉토리")
    parser.add_argument("--prompt", type=str, help="CLIP Score 계산용 텍스트 프롬프트")
    parser.add_argument("--output", type=str, help="결과 CSV 저장 경로")
    parser.add_argument("--device", type=str, default="auto", help="cuda / cpu / auto (기본: auto)")
    args = parser.parse_args()

    if not args.images and not args.images_dir:
        parser.error("--images 또는 --images-dir 중 하나는 필요")
    if not args.base and not args.prompt:
        parser.error("--base (LPIPS) 또는 --prompt (CLIP Score) 중 하나는 필요")

    # 디바이스
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"device: {device}")

    # 이미지 수집
    compare_paths = collect_images(args.images, args.images_dir)
    if not compare_paths:
        print("[오류] 평가할 이미지가 없음", file=sys.stderr)
        sys.exit(1)

    compare_imgs = [load_image(p) for p in compare_paths]

    # base image에서 compare_paths에 있으면 제외 (디렉토리 모드에서 base가 섞이는 경우)
    if args.base:
        base_path = Path(args.base).resolve()
        filtered = [(p, img) for p, img in zip(compare_paths, compare_imgs) if p.resolve() != base_path]
        if len(filtered) != len(compare_paths):
            print(f"[정보] base image를 비교 대상에서 제외: {base_path.name}")
        compare_paths, compare_imgs = zip(*filtered) if filtered else ([], [])
        compare_paths, compare_imgs = list(compare_paths), list(compare_imgs)

    if not compare_imgs:
        print("[오류] 비교할 이미지가 없음", file=sys.stderr)
        sys.exit(1)

    rows = [{"image": p.name, "lpips": None, "clip_score": None} for p in compare_paths]

    # LPIPS
    if args.base:
        base_path = Path(args.base)
        if not base_path.exists():
            print(f"[오류] base image 없음: {base_path}", file=sys.stderr)
            sys.exit(1)
        base_img = load_image(base_path)
        print(f"LPIPS 계산 중 (base: {base_path.name}) ...")
        lpips_scores = compute_lpips(base_img, compare_imgs, device)
        for row, score in zip(rows, lpips_scores):
            row["lpips"] = round(score, 4)

    # CLIP Score
    if args.prompt:
        print(f"CLIP Score 계산 중 (prompt: \"{args.prompt}\") ...")
        clip_scores = compute_clip_scores(compare_imgs, args.prompt, device)
        for row, score in zip(rows, clip_scores):
            row["clip_score"] = score

    print_results(rows)

    if args.output:
        save_csv(rows, args.output)


if __name__ == "__main__":
    main()
