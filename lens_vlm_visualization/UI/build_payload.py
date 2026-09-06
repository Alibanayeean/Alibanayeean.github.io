"""Builds the data behind the VLM patch explorer UI (see README.md).

Two datasets, 20 images each:

  coco     -- mtool_coco_instance_segmentations.json: every object instance carries a
              `category`, a `synonyms` list and a `mask_path`. A patch's SEMANTICALLY
              RELATED words are the single-token variants of the synonyms of whichever
              object masks cover that patch -- so the mask, not a guess, decides what is
              related, exactly as requested. A patch covered by no mask has no related
              words, and the UI says so rather than inventing them.
  counting  -- the CLEVR black-spheres scenes. Here the target is a property of the whole
              scene, not of any one patch, so every patch gets the SAME word set: the
              digit and word forms of the numbers one..nine, filtered to exactly-one-token
              variants. Every one of those is highlighted in the top-10 lists.

Word resolution everywhere is the project's exact-single-token rule -- a surface form
that needs more than one token is DROPPED, never approximated by its first sub-word
fragment (which would match countless unrelated words). Reused unchanged from
../new_coco/targets.py and ../counterfactual_counting_clever_experiments/targets.py.

For every patch of every image, for all four methods (logit_lens, r_lens, j_lens,
plain_pip) and every layer that method covers, this stores:

  top  -- the 10 highest-probability vocabulary tokens, as [vocab_index, probability]
  cen  -- the CENTERED LOGIT of each related word: logit - mean logit over the whole
          248,320-token vocabulary, so 0 means exactly "the average vocabulary entry".
          Same definition as qwen_3_5_9b/llm/final_report_results/centered_metrics.py.

Storage notes: token strings are pooled into a per-image `vocab` table and referenced by
index (a handful of tokens dominate top-1 across most patches), probabilities are rounded
to 4 dp and centered logits to 1 dp -- both far finer than the UI displays. One JSON file
per patch, so the browser only ever fetches the patch you clicked.

Usage:
    .venv/bin/python qwen_3_5_9b/vlm/UI/build_payload.py
    .venv/bin/python qwen_3_5_9b/vlm/UI/build_payload.py --datasets counting --limit 1
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent          # qwen_3_5_9b/vlm/UI
VLM_ROOT = ROOT.parent
QWEN35_ROOT = VLM_ROOT.parent
MTOOL_ROOT = QWEN35_ROOT.parent
QWEN38B_ROOT = MTOOL_ROOT / "qwen_3_8b"
for root in (QWEN35_ROOT, QWEN38B_ROOT, MTOOL_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from pip_qwen35.lenses import load_lenses  # noqa: E402
from pip_qwen35.model_utils import load_qwen35, model_device, model_display_name  # noqa: E402
from qwen_3_5_9b.vlm.interventions import capture_multimodal_traces  # noqa: E402
from qwen_3_5_9b.vlm.run_experiment import _sub_trace  # noqa: E402
from qwen_3_5_9b.vlm.vision_utils import (  # noqa: E402
    PreparedImage, load_annotations, IMAGES_DIR as COCO_IMAGES_DIR,
    VISION_START_TOKEN_ID, VISION_END_TOKEN_ID, IMAGE_TOKEN_ID,
)
from qwen_3_5_9b.vlm.new_coco import metrics as nc  # noqa: E402
from qwen_3_5_9b.vlm.new_coco.targets import variants, single_token_id_or_none  # noqa: E402
from qwen_3_5_9b.vlm.counterfactual_counting_clever_experiments import dataset as counting_dataset  # noqa: E402
from qwen_3_5_9b.vlm.counterfactual_counting_clever_experiments.targets import NUMBER_WORDS  # noqa: E402

R_LENS_PATH = "/home/mmd/ali/workspace-lenses/qwen3.5-9b/r-lens/lens.pt"
J_LENS_PATH = "/home/mmd/ali/workspace-lenses/qwen3.5-9b/j-lens/lens.pt"
NUM_LAYERS = 32
TOP_K = 10
DEFAULT_BATCH_SIZE = 48
NUM_IMAGES = 20
DATA_DIR = ROOT / "data"
IMAGES_OUT = ROOT / "images"
METHODS = ("plain_pip", "logit_lens", "j_lens", "r_lens")   # display order: ours first
METHOD_LABELS = {
    "plain_pip": "Ours",
    "logit_lens": "Logit lens",
    "j_lens": "J-lens",
    "r_lens": "R-lens",
}
# The model is given NO text prompt in any of these experiments -- the input really is
# just the image, wrapped in its vision sentinels. Displayed verbatim in the UI.
PROMPT_NOTE = ("There is no text prompt. The model is never asked a question: the entire "
               "input is the image itself, wrapped in vision sentinels. Every readout below "
               "is taken at a VISUAL token (an image patch), not at a text position.")


def _word_entries(tokenizer: Any, surfaces: list[str]) -> list[dict[str, Any]]:
    """[{surface, id}] for the surfaces that tokenize to exactly one token."""
    out, seen = [], set()
    for surface in surfaces:
        token_id = single_token_id_or_none(tokenizer, surface)
        if token_id is None or token_id in seen:
            continue
        seen.add(token_id)
        out.append({"surface": surface, "id": token_id})
    return out


def coco_images(tokenizer: Any, limit: int) -> list[dict[str, Any]]:
    """First `limit` images by file name -- a deterministic, unbiased selection rather
    than a cherry-pick (COCO images here are uniform: 3-5 annotated objects each)."""
    annotations = load_annotations()
    out = []
    for name in sorted(annotations)[:limit]:
        meta = annotations[name]
        groups = []
        for inst in meta["instances"]:
            surfaces = [form for syn in inst["synonyms"] for form in variants(syn)]
            words = _word_entries(tokenizer, surfaces)
            groups.append({
                "key": f"instance{inst['instance_number']}",
                "label": inst["category"],
                "instance_number": inst["instance_number"],
                "mask_path": inst["mask_path"],
                "words": words,
                "num_dropped": len({f for f in surfaces}) - len(words),
            })
        out.append({"image_name": name, "meta": meta, "groups": groups})
    return out


def counting_images(tokenizer: Any, limit: int) -> list[dict[str, Any]]:
    """All numbers one..nine, digit and word forms, single-token variants only. The count
    is a property of the whole scene, so every patch shares this word set."""
    scenes = counting_dataset.load_scenes()
    groups = []
    for n in range(1, 10):
        surfaces = variants(str(n)) + variants(NUMBER_WORDS[n])
        groups.append({"key": f"n{n}", "label": f"{n} / \"{NUMBER_WORDS[n]}\"",
                       "number": n, "mask_path": None,
                       "words": _word_entries(tokenizer, surfaces), "num_dropped": 0})
    out = []
    for name in sorted(scenes)[:limit]:
        out.append({"image_name": name, "meta": scenes[name], "groups": groups,
                    "true_count": scenes[name]["count"]})
    return out


@torch.inference_mode()
def build_image(model, tokenizer, r_lens, j_lens, row_norms, bias, entry, *,
                dataset_key: str, batch_size: int, out_dir: Path) -> dict[str, Any]:
    name = entry["image_name"]
    images_dir = COCO_IMAGES_DIR if dataset_key == "coco" else counting_dataset.IMAGES_DIR
    prepared = PreparedImage(name, entry["meta"], images_dir=images_dir)
    device = model_device(model)
    num_patches = prepared.num_visual_tokens

    # Which word-groups are semantically related to each patch. COCO: decided by the
    # object masks. Counting: the scene-level number set applies to every patch.
    mask_root = None if dataset_key == "coco" else counting_dataset.DATASET_ROOT
    groups = entry["groups"]
    patch_groups: list[list[int]] = [[] for _ in range(num_patches)]
    for gi, group in enumerate(groups):
        if group["mask_path"] is None:
            for p in range(num_patches):
                patch_groups[p].append(gi)
            continue
        for p in prepared.mask_patch_indices(group["mask_path"], mask_root=mask_root):
            if p < num_patches:
                patch_groups[p].append(gi)

    # Every word id that any group of this image can ask about -- one readout column set
    # per image, sliced per patch afterwards.
    all_ids: list[int] = []
    id_index: dict[int, int] = {}
    for group in groups:
        for w in group["words"]:
            if w["id"] not in id_index:
                id_index[w["id"]] = len(all_ids)
                all_ids.append(w["id"])
        group["word_cols"] = [id_index[w["id"]] for w in group["words"]]
    target_ids = torch.tensor(all_ids, device=device, dtype=torch.long)

    print(f"{dataset_key}/{name}: {num_patches} patches, {len(groups)} groups, {len(all_ids)} word ids")
    trace = capture_multimodal_traces(
        model, input_ids=prepared.input_ids, pixel_values=prepared.pixel_values,
        image_grid_thw=prepared.image_grid_thw, mm_token_type_ids=prepared.mm_token_type_ids,
        target_positions=prepared.visual_token_positions,
    )

    layers_by_method = {
        "logit_lens": list(range(1, NUM_LAYERS + 1)),
        "plain_pip": list(range(1, NUM_LAYERS + 1)),
        "r_lens": sorted(l + 1 for l in r_lens.source_layers),
        "j_lens": sorted(l + 1 for l in j_lens.source_layers),
    }
    # Each patch only ever displays the words of the groups that cover it, so only those
    # columns are stored. A COCO patch outside every mask therefore carries no centered
    # logits at all (and the UI says so), which is what keeps the payload small: storing
    # all of an image's word columns at every patch made one 192-patch image 22 MB.
    patch_cols: list[list[int]] = []
    for p in range(num_patches):
        cols: list[int] = []
        seen: set[int] = set()
        for gi in patch_groups[p]:
            for c in groups[gi]["word_cols"]:
                if c not in seen:
                    seen.add(c)
                    cols.append(c)
        patch_cols.append(cols)

    # per patch -> per method -> per layer -> {"top": [[vocab_idx, prob]], "cen": [...]}
    per_patch: list[dict[str, dict[str, Any]]] = [{m: {} for m in METHODS} for _ in range(num_patches)]
    vocab: list[list[Any]] = []      # [surface, token_id]
    vocab_index: dict[int, int] = {}

    t0 = time.time()
    current_batch_size = batch_size
    start = 0
    while start < num_patches:
        end = min(start + current_batch_size, num_patches)
        indices = torch.arange(start, end, device=device)
        try:
            sub = _sub_trace(trace, indices)
            for method, layer, state in nc.method_states(model, sub, r_lens, j_lens, num_layers=NUM_LAYERS):
                logits = model.lm_head(state).float()
                probs = torch.softmax(logits, dim=-1)
                top_p, top_i = probs.topk(TOP_K, dim=-1)
                dots = logits if bias is None else logits - bias
                centered = dots.index_select(1, target_ids) - dots.mean(dim=-1, keepdim=True)
                top_p_np = top_p.cpu().numpy(); top_i_np = top_i.cpu().numpy()
                cen_np = centered.cpu().numpy()
                for row in range(end - start):
                    tops = []
                    for rank in range(TOP_K):
                        tid = int(top_i_np[row, rank])
                        if tid not in vocab_index:
                            vocab_index[tid] = len(vocab)
                            vocab.append([tokenizer.decode([tid]), tid])
                        tops.append([vocab_index[tid], round(float(top_p_np[row, rank]), 4)])
                    cols = patch_cols[start + row]
                    entry: dict[str, Any] = {"top": tops}
                    if cols:
                        entry["cen"] = [round(float(cen_np[row, c]), 1) for c in cols]
                    per_patch[start + row][method][str(layer)] = entry
                del logits, probs, top_p, top_i, dots, centered
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            if current_batch_size <= 4:
                raise
            current_batch_size = max(4, current_batch_size // 2)
            print(f"  OOM at [{start}:{end}] -- retrying with batch_size={current_batch_size}")
            continue
        torch.cuda.empty_cache()
        print(f"  patches [{start}:{end}] of {num_patches} ({time.time() - t0:.0f}s)")
        start = end

    stem = Path(name).stem
    image_dir = out_dir / stem
    (image_dir / "patches").mkdir(parents=True, exist_ok=True)
    for p in range(num_patches):
        (image_dir / "patches" / f"{p}.json").write_text(
            json.dumps({"patch": p, "cols": patch_cols[p], "methods": per_patch[p]},
                       separators=(",", ":")), encoding="utf-8")

    IMAGES_OUT.joinpath(dataset_key).mkdir(parents=True, exist_ok=True)
    shutil.copyfile(images_dir / name, IMAGES_OUT / dataset_key / name)

    prompt_tokens = [
        {"text": "<|vision_start|>", "kind": "sentinel"},
        {"text": f"<|image_pad|> × {num_patches}", "kind": "image"},
        {"text": "<|vision_end|>", "kind": "sentinel"},
    ]
    index = {
        "dataset": dataset_key,
        "image_name": name,
        "image_file": f"images/{dataset_key}/{name}",
        "grid_h": prepared.grid_h, "grid_w": prepared.grid_w,
        "num_patches": num_patches,
        "orig_width": prepared.orig_width, "orig_height": prepared.orig_height,
        "layers_by_method": layers_by_method,
        "methods": list(METHODS), "method_labels": METHOD_LABELS,
        "vocab": vocab,
        "groups": [{k: v for k, v in g.items() if k != "mask_path"} for g in groups],
        "patch_groups": patch_groups,
        "prompt_tokens": prompt_tokens,
        "prompt_note": PROMPT_NOTE,
        "sequence_length": num_patches + 2,
        "true_count": entry.get("true_count"),
    }
    (image_dir / "index.json").write_text(json.dumps(index, separators=(",", ":")), encoding="utf-8")
    size = sum(f.stat().st_size for f in image_dir.rglob("*.json")) / 1e6
    print(f"  wrote {image_dir} ({size:.1f} MB)")
    return {"stem": stem, "image_name": name, "num_patches": num_patches,
            "num_groups": len(groups), "dir": f"data/{dataset_key}/{stem}"}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16", "float32"))
    parser.add_argument("--datasets", default="coco,counting")
    parser.add_argument("--limit", type=int, default=NUM_IMAGES)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    wanted = [d.strip() for d in args.datasets.split(",")]
    model, tokenizer, resolved = load_qwen35(args.model_path, device=args.device, dtype=args.dtype)
    print(f"Loaded {model_display_name(model, resolved)} from {resolved}")
    r_lens, j_lens = load_lenses(r_lens_path=R_LENS_PATH, j_lens_path=J_LENS_PATH)
    bias = nc.lm_head_bias(model)
    row_norms = None
    print(f"lm_head bias: {'present (subtracted)' if bias is not None else 'none'}")

    datasets_meta = []
    for key in wanted:
        entries = (coco_images if key == "coco" else counting_images)(tokenizer, args.limit)
        out_dir = DATA_DIR / key
        out_dir.mkdir(parents=True, exist_ok=True)
        images = []
        for entry in entries:
            stem = Path(entry["image_name"]).stem
            if (out_dir / stem / "index.json").is_file() and not args.overwrite:
                existing = json.loads((out_dir / stem / "index.json").read_text())
                images.append({"stem": stem, "image_name": entry["image_name"],
                               "num_patches": existing["num_patches"],
                               "num_groups": len(existing["groups"]), "dir": f"data/{key}/{stem}"})
                print(f"{key}/{entry['image_name']}: already built, skipping")
                continue
            images.append(build_image(model, tokenizer, r_lens, j_lens, row_norms, bias, entry,
                                      dataset_key=key, batch_size=args.batch_size, out_dir=out_dir))
        datasets_meta.append({
            "key": key,
            "title": "COCO (objects)" if key == "coco" else "CLEVR counting (black spheres)",
            "highlight_rule": ("Words of the object(s) whose mask covers the selected patch."
                               if key == "coco" else
                               "All single-token variants of the numbers one..nine (the scene-level target)."),
            "images": images,
        })

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "index.json").write_text(json.dumps({
        "datasets": datasets_meta, "methods": list(METHODS), "method_labels": METHOD_LABELS,
        "top_k": TOP_K, "prompt_note": PROMPT_NOTE,
    }, indent=1), encoding="utf-8")
    print(f"wrote {DATA_DIR / 'index.json'}")


if __name__ == "__main__":
    main()
