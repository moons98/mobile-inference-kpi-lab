#!/usr/bin/env python3
"""
Phase 2 이미지 품질 평가 — LPIPS + CLIP Score.
base image: B2 (SD v1.5 50 steps, mixed_pr + vae w8a8) — phase 2 quality 상한선.
결과를 txt 테이블로 outputs/exp/ 에 저장.

B3: LCM s8 vae w8a8.
"""
import torch
from PIL import Image
from pathlib import Path
from datetime import datetime
import lpips
import torchvision.transforms.functional as TF
from transformers import CLIPProcessor, CLIPModel

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "exp"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PROMPT = "a photo of a cat sitting on a windowsill, watercolor painting, soft edges, delicate, artistic"

# (ID, label, image_path, base_id)
# base_id: LPIPS 기준 이미지 ID (B2 = SD 50step, quality 상한선)
ENTRIES = [
    ("A1", "SD  s20  mixed_pr+w8a8 (phase1)",  "logs/phase1/txt2img_sd15_mixed_tfp16_umixed-pr_vw8a8_qnn_npu_s20_single_20260313_195034_last.png", "B2"),
    ("B1", "SD  s30  mixed_pr+w8a8",            "logs/phase2/txt2img_sd15_mixed_tfp16_umixed-pr_vw8a8_qnn_npu_s30_single_20260313_210149_last.png",  "B2"),
    ("B2", "SD  s50  mixed_pr+w8a8 (base)",     "logs/phase2/txt2img_sd15_mixed_tfp16_umixed-pr_vw8a8_qnn_npu_s50_single_20260313_211043_last.png",  "B2"),
    ("A5", "LCM s4   fp16 (phase1)",            "logs/phase1/txt2img_lcm_fp16_qnn_npu_s4_single_20260313_195823_last.png",                           "B2"),
    ("B3", "LCM s8   vae w8a8",                  "logs/phase2/txt2img_lcm_mixed_tfp16_ufp16_vw8a8_qnn_npu_s8_single_20260313_212940_last.png",        "B2"),
]

# LCM 내부 step sweep 비교 (B3 기준 — intra-LCM)
LCM_ENTRIES = [
    ("A5", "LCM s4  fp16  vs B3 (LCM base)", "logs/phase1/txt2img_lcm_fp16_qnn_npu_s4_single_20260313_195823_last.png",          "B3"),
    ("B3", "LCM s8  vae w8a8  (LCM base)",    "logs/phase2/txt2img_lcm_mixed_tfp16_ufp16_vw8a8_qnn_npu_s8_single_20260313_212940_last.png", "B3"),
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")

# --- LPIPS ---
print("LPIPS 계산 중...")
loss_fn = lpips.LPIPS(net="alex").to(device)
loss_fn.eval()

images = {}
for eid, label, path, base_id in ENTRIES:
    images[eid] = Image.open(PROJECT_ROOT / path).convert("RGB")

def to_lpips_t(img):
    t = TF.to_tensor(img) * 2.0 - 1.0
    return t.unsqueeze(0).to(device)

base_b2 = images["B2"]
base_b3 = images["B3"]
bases = {"B2": base_b2, "B3": base_b3}

lpips_scores = {}
with torch.no_grad():
    for eid, label, path, base_id in ENTRIES:
        base = bases[base_id]
        img = images[eid]
        if img.size != base.size:
            img = img.resize(base.size, Image.BICUBIC)
        lpips_scores[eid] = loss_fn(to_lpips_t(base), to_lpips_t(img)).item()

lcm_lpips = {}
with torch.no_grad():
    for eid, label, path, base_id in LCM_ENTRIES:
        base = bases[base_id]
        img = images[eid]
        if img.size != base.size:
            img = img.resize(base.size, Image.BICUBIC)
        lcm_lpips[eid] = loss_fn(to_lpips_t(base), to_lpips_t(img)).item()

# --- CLIP Score ---
print("CLIP Score 계산 중...")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
clip_proc  = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
clip_model.eval()

clip_scores = {}
with torch.no_grad():
    for eid, label, path, base_id in ENTRIES:
        inputs = clip_proc(text=[PROMPT], images=images[eid], return_tensors="pt", padding=True)
        outputs = clip_model(**inputs)
        clip_scores[eid] = outputs.logits_per_image.item()

# --- 출력 ---
W = 100
sep  = "=" * W
sep2 = "-" * W

lines = []
lines.append(sep)
lines.append("Phase 2 — Step Sweep Image Quality Evaluation")
lines.append(sep)
lines.append("")
lines.append(f"  Prompt: \"{PROMPT}\"")
lines.append(f"  LPIPS base: B2 = SD v1.5 s50 mixed_pr+vae_w8a8  (quality 상한선)")
lines.append(f"  CLIP Score: openai/clip-vit-base-patch16  (logit scale, ↑ better)")
lines.append(f"  LPIPS:      AlexNet perceptual distance vs B2  (↓ better, 0 = identical)")
lines.append(f"  LCM base:   A5 = s4 fp16 (Phase 1) / B3 = s8 vae w8a8")
lines.append("")
lines.append(sep2)
lines.append(f"  {'ID':<4}  {'Config':<32}  {'CLIP Score ↑':>14}  {'LPIPS ↓':>10}  {'vs base':>6}")
lines.append(f"  {'----':<4}  {'--------------------------------':<32}  {'-------------':>14}  {'-------':>10}  {'------':>6}")
for eid, label, path, base_id in ENTRIES:
    is_base = (eid == base_id)
    lpips_str = "—  (base)" if is_base else f"{lpips_scores[eid]:.4f}"
    lines.append(f"  {eid:<4}  {label:<32}  {clip_scores[eid]:>14.4f}  {lpips_str:>10}")
lines.append(sep2)
lines.append("")
lines.append("  [LCM 내부 step sweep 비교]  LPIPS base = B3 (LCM s8 fp16)")
lines.append("")
lines.append(f"  {'ID':<4}  {'Config':<32}  {'CLIP Score ↑':>14}  {'LPIPS ↓':>10}")
lines.append(f"  {'----':<4}  {'--------------------------------':<32}  {'-------------':>14}  {'-------':>10}")
for eid, label, path, base_id in LCM_ENTRIES:
    is_base = (eid == base_id)
    lpips_str = "—  (base)" if is_base else f"{lcm_lpips[eid]:.4f}"
    lines.append(f"  {eid:<4}  {label:<32}  {clip_scores[eid]:>14.4f}  {lpips_str:>10}")
lines.append(sep2)
lines.append(f"  SD  : A1 CLIP={clip_scores['A1']:.4f}  B1={clip_scores['B1']:.4f}  B2={clip_scores['B2']:.4f}")
lines.append(f"  LCM : A5 CLIP={clip_scores['A5']:.4f}  B3={clip_scores['B3']:.4f}")
lines.append(sep2)
lines.append("")
lines.append(sep)
lines.append("")
lines.append("Interpretation Notes")
lines.append(sep2)
lines.append("  [SEED=42 고정 전제 — SD 내부 step sweep]")
lines.append("    A1/B1/B2 모두 동일 precision(mixed_pr+w8a8), 동일 seed → z_T 동일.")
lines.append("    → LPIPS 차이는 step 수 변화에 의한 denoising quality 차이만 반영.")
lines.append("    step 수 증가 → B2(50step)에 가까워질수록 LPIPS 감소 기대.")
lines.append("")
lines.append("  [LCM vs SD 비교 — B2 기준 LPIPS]")
lines.append("    LCM과 SD는 UNet 구조·scheduler가 달라 z_T가 같아도 trajectory가 크게 분기.")
lines.append("    → LPIPS는 '품질 격차' 지표로 해석. CLIP Score 절대값으로 prompt 반영도 확인.")
lines.append("")
lines.append("  [CLIP Score — openai/clip-vit-base-patch16, logit scale]")
lines.append("    > 33   : 양호")
lines.append("    27~33  : 보통 (조건부)")
lines.append("    < 27   : 탈락")
lines.append("    * ±2 이내 차이는 유의미하지 않음")
lines.append("")
lines.append("  [LPIPS — AlexNet, [-1,1] 정규화 입력]")
lines.append("    < 0.05 : 사실상 동일")
lines.append("    0.05~0.20 : 미세한 색감·텍스처 차이")
lines.append("    0.20~0.50 : 구도/스타일은 유사하나 세부 내용 다름")
lines.append("    0.50~0.70 : 다른 장면 → CLIP으로 판정")
lines.append("    > 0.70 : 완전히 다른 이미지")
lines.append(sep2)

report = "\n".join(lines)
print("\n" + report)

ts = datetime.now().strftime("%Y%m%d_%H%M%S")
out_path = OUTPUT_DIR / f"quality_phase2_{ts}.txt"
out_path.write_text(report, encoding="utf-8")
print(f"\nSaved to {out_path}")
