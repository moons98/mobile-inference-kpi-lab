#!/usr/bin/env python3
"""
Phase 1 이미지 품질 평가 — LPIPS + CLIP Score.
결과를 txt 테이블로 outputs/exp/ 에 저장.
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
# base_id: LPIPS 기준 이미지 ID
ENTRIES = [
    ("A1", "SD v1.5  fp16 (SD base)",     "logs/phase1/txt2img_sd15_fp16_qnn_npu_s20_single_20260313_184651_last.png",                         "A1"),
    ("A2", "SD v1.5  unet mixed_pr",      "logs/phase1/txt2img_sd15_mixed_tfp16_umixed-pr_vfp16_qnn_npu_s20_single_20260313_193258_last.png",  "A1"),
    ("A3", "SD v1.5  vae w8a8",           "logs/phase1/txt2img_sd15_mixed_tfp16_ufp16_vw8a8_qnn_npu_s20_single_20260313_194200_last.png",      "A1"),
    ("A4", "SD v1.5  mixed_pr + vae w8a8","logs/phase1/txt2img_sd15_mixed_tfp16_umixed-pr_vw8a8_qnn_npu_s20_single_20260313_195034_last.png",  "A1"),
    ("A5", "LCM      fp16 (LCM base)",    "logs/phase1/txt2img_lcm_fp16_qnn_npu_s4_single_20260313_195823_last.png",                           "A5"),
    ("A6", "LCM      vae w8a8",           "logs/phase1/txt2img_lcm_mixed_tfp16_ufp16_vw8a8_qnn_npu_s4_single_20260313_200423_last.png",        "A5"),
]

# LCM vs SD 직접 비교 (A1 기준으로 A5 LPIPS 추가)
CROSS_ENTRIES = [
    ("A5", "LCM fp16    vs SD fp16 (A1 base)", "logs/phase1/txt2img_lcm_fp16_qnn_npu_s4_single_20260313_195823_last.png",                          "A1"),
    ("A6", "LCM vae w8a8 vs SD fp16 (A1 base)", "logs/phase1/txt2img_lcm_mixed_tfp16_ufp16_vw8a8_qnn_npu_s4_single_20260313_200423_last.png",  "A1"),
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

lpips_scores = {}
bases = {"A1": images["A1"], "A5": images["A5"]}
with torch.no_grad():
    for eid, label, path, base_id in ENTRIES:
        base = bases[base_id]
        img = images[eid]
        if img.size != base.size:
            img = img.resize(base.size, Image.BICUBIC)
        score = loss_fn(to_lpips_t(base), to_lpips_t(img)).item()
        lpips_scores[eid] = score

# LCM vs SD cross LPIPS
cross_lpips = {}
with torch.no_grad():
    for eid, label, path, base_id in CROSS_ENTRIES:
        base = bases[base_id]
        img = images[eid]
        if img.size != base.size:
            img = img.resize(base.size, Image.BICUBIC)
        score = loss_fn(to_lpips_t(base), to_lpips_t(img)).item()
        cross_lpips[eid] = score

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
W = 90
sep  = "=" * W
sep2 = "-" * W

lines = []
lines.append(sep)
lines.append("Phase 1 — Image Quality Evaluation")
lines.append(sep)
lines.append("")
lines.append(f"  Prompt: \"{PROMPT}\"")
lines.append(f"  LPIPS base: A1 = SD v1.5 fp16  |  A5 = LCM fp16")
lines.append(f"  CLIP Score: openai/clip-vit-base-patch16  (logit scale, ↑ better)")
lines.append(f"  LPIPS:      AlexNet perceptual distance vs base  (↓ better, 0 = identical)")
lines.append("")
lines.append(sep2)
lines.append(f"  {'ID':<4}  {'Config':<28}  {'CLIP Score ↑':>14}  {'LPIPS ↓':>10}  {'vs base':>6}")
lines.append(f"  {'----':<4}  {'----------------------------':<28}  {'-------------':>14}  {'-------':>10}  {'------':>6}")
for eid, label, path, base_id in ENTRIES:
    is_base = (eid == base_id)
    lpips_str = "—  (base)" if is_base else f"{lpips_scores[eid]:.4f}"
    lines.append(f"  {eid:<4}  {label:<28}  {clip_scores[eid]:>14.4f}  {lpips_str:>10}")
lines.append(sep2)
lines.append("")
lines.append("  [SD vs LCM 직접 비교]  LPIPS base = A1 (SD v1.5 fp16)")
lines.append("")
lines.append(f"  {'ID':<4}  {'Config':<28}  {'CLIP Score ↑':>14}  {'LPIPS ↓':>10}")
lines.append(f"  {'----':<4}  {'----------------------------':<28}  {'-------------':>14}  {'-------':>10}")
for eid, label, path, base_id in CROSS_ENTRIES:
    lines.append(f"  {eid:<4}  {label:<28}  {clip_scores[eid]:>14.4f}  {cross_lpips[eid]:>10.4f}")
lines.append(sep2)
lines.append(f"  A1 CLIP={clip_scores['A1']:.4f}  |  A5 CLIP={clip_scores['A5']:.4f} (차이 {clip_scores['A5']-clip_scores['A1']:+.4f})  |  A6 CLIP={clip_scores['A6']:.4f} (차이 {clip_scores['A6']-clip_scores['A1']:+.4f})")
lines.append(sep2)
lines.append("")
lines.append(sep)
lines.append("")
lines.append("Interpretation Notes")
lines.append(sep2)
lines.append("  [LPIPS 비교의 전제 — 왜 경량화 모델 간 LPIPS 비교가 유효한가]")
lines.append("    모든 실험은 SEED=42 고정 (java.util.Random(42), Box-Muller 변환).")
lines.append("    → 모든 config가 동일한 초기 noise z_T 에서 출발하므로 최종 이미지 차이는")
lines.append("      순전히 모델(양자화)의 영향이며, LPIPS가 그 차이를 직접 측정한다.")
lines.append("    VAE w8a8  : UNet 동일 → denoising trajectory 동일 → z_0 동일")
lines.append("                → LPIPS = VAE decode 품질 차이만 반영  (기대값: 낮음)")
lines.append("    UNet mixed_pr : step마다 양자화 오차 누적 → z_0 자체가 달라짐")
lines.append("                → LPIPS = trajectory 분기 반영  (기대값: 높음)")
lines.append("    ∴ VAE w8a8 LPIPS ≪ UNet mixed_pr LPIPS 는 설계 의도와 일치하는 내적 일관성.")
lines.append("")
lines.append("  [CLIP Score — openai/clip-vit-base-patch16, logit scale]")
lines.append("    < 20   : 이미지와 프롬프트 거의 무관")
lines.append("    20~27  : 프롬프트를 약하게 반영 (구도/피사체 대략 맞음)")
lines.append("    27~33  : 보통 수준 — 주요 요소는 있으나 세부 묘사 부족")
lines.append("    33~38  : 양호 — 프롬프트 핵심 요소를 잘 반영")
lines.append("    > 38   : 매우 높음 (사진-캡션 쌍 수준)")
lines.append("    * 동일 프롬프트 내 비교 시 ±1 이내 차이는 유의미하지 않음")
lines.append("    * 모델·스텝 수가 다른 그룹 간 비교(SD vs LCM)에는 절대값보다 상대 차이에 주목")
lines.append("")
lines.append("  [LPIPS — AlexNet, [-1,1] 정규화 입력]")
lines.append("    < 0.05 : 사실상 동일 (압축 아티팩트 수준의 차이)")
lines.append("    0.05~0.20 : 미세한 색감·텍스처 차이, 육안 구분 어려움")
lines.append("    0.20~0.50 : 구도/스타일은 유사하나 세부 내용 다름")
lines.append("    0.50~0.70 : 같은 프롬프트지만 다른 장면처럼 보임 (UNet 경로 변경 등)")
lines.append("    > 0.70 : 완전히 다른 이미지")
lines.append("    * UNet 양자화(mixed_pr)는 denoising trajectory 자체를 바꾸므로 LPIPS가 높아도")
lines.append("      CLIP score가 유지되면 '다른 이미지지만 품질은 동등'으로 해석 가능")
lines.append("    * VAE 양자화(w8a8)는 동일 latent를 디코딩하므로 LPIPS가 낮게 나오는 것이 정상")
lines.append(sep2)

report = "\n".join(lines)
print("\n" + report)

ts = datetime.now().strftime("%Y%m%d_%H%M%S")
out_path = OUTPUT_DIR / f"quality_phase1_{ts}.txt"
out_path.write_text(report, encoding="utf-8")
print(f"\nSaved to {out_path}")
