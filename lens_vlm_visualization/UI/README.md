# VLM patch explorer

A static site for reading the four decoders at a **single image patch**, on 20 COCO
images and 20 CLEVR counting scenes. Serve it and open the page:

```bash
cd qwen_3_5_9b/vlm/UI && python3 -m http.server
```

Modelled on `qwen_3_5_9b/llm/new_datasets/UI` (same two-view layout, same "ours first"
column order, same per-example fetching), with the text prompt replaced by a clickable
patch grid.

## What it shows

Pick a dataset, pick an image, then **click a patch**. Two views over that patch:

| view | contents |
|---|---|
| **Top-10 tokens** | per layer and method, the 10 highest-probability vocabulary tokens with their probabilities. A token is **highlighted green** when it is semantically related to the selected patch (rule below). |
| **Centered logits** | per layer and method, the centered logit of every word related to that patch — its logit minus the mean logit over the whole 248,320-token vocabulary, so **0 means "exactly the average vocabulary entry"**. Signed: green positive, red negative. |

Columns are **Ours (PIP) first**, then logit lens, J-lens, R-lens. Layers run 32 → 1.
R-lens and J-lens have no layer-32 translator (this project's standing convention), so
their layer-32 cell reads "no translator" rather than being blank.

Hovering an object label tints every patch that object's mask covers, so you can see what
the annotation actually claims before trusting a highlight.

## What counts as "semantically related" — decided by the annotations, not by guesswork

**COCO.** A patch's related words are the words of whichever object **masks cover that
patch**, taken from `mtool_coco_instance_segmentations.json`: each instance's `category`
plus its `synonyms`, expanded through the usual case/leading-space variant grid. Mask →
patch coverage uses `vision_utils.mask_patch_indices`, the same function the mask-based
experiments use, so the UI and the metrics agree on which patches belong to an object.

A patch that **no** mask covers has no related words. The UI says so ("no annotated object
covers this patch — nothing to highlight") and the centered-logit view is empty for it,
rather than inventing a relation. This is common: most patches of a COCO image are
background.

**Counting.** The target is a property of the whole scene (how many spheres it has), not
of any one patch, so every patch shares the same word set: the numbers **one … nine**, in
both digit and word form. All of them are highlighted in the top-10 lists, per request.

**Word resolution is exact-single-token everywhere.** A surface form that needs more than
one token is **dropped**, never approximated by its first sub-word fragment — that
fragment would match countless unrelated words and register as a false highlight. Reused
from `../new_coco/targets.py`. This drops a lot (`"Oven"` → `['O','ven']`, `"stovetop"` →
4 tokens), which is why an object may show fewer words than its synonym list suggests.

## The prompt

Shown at the top of every image, and worth reading once: **there is no text prompt.** The
model is never asked a question. The entire input is

```
<|vision_start|>   <|image_pad|> × N   <|vision_end|>
```

— 194 tokens for a 192-patch COCO image, 302 for a 300-patch counting scene, with zero
text tokens. Every readout in this UI is taken at a **visual** token.

That matters for interpretation, especially on the counting set: a number in the top-10 at
a patch means that patch's representation locally promotes that number token, **not** that
the model counted the spheres. Nothing here asks it to count, and the counting
experiment's own distractor control shows the wrong count scoring 84% of the right count's
confidence under PIP (and higher on the `max` reduction).

## The methods

The same four used throughout the VLM work, computed by
`../new_coco/metrics.py`'s `method_states` — the exact pre-`lm_head` vector each reported
metric is derived from, so these panels cannot disagree with the curves in the experiment
folders:

- **Ours (PIP)** — self-only isolation from the cut layer onward, α=β=γ=1, decoded once.
- **Logit lens** — the real residual, read out through the real unembedding.
- **R-lens / J-lens** — the externally-fitted linear translators, layers 1–31 only.

At layer 32 PIP applies no isolation at all, so its column is identical to the logit lens
there — the endpoint invariant, visible live in the table.

## Image selection

The **first 20 images by file name** in each dataset — deterministic and unbiased, not
cherry-picked. (COCO images here are uniform: 3–5 annotated objects each. The counting set
has exactly 20 scenes, so all of them are included.)

## Building the data

```bash
.venv/bin/python qwen_3_5_9b/vlm/UI/build_payload.py                       # both datasets, 20 each
.venv/bin/python qwen_3_5_9b/vlm/UI/build_payload.py --datasets counting --limit 1   # spot check
```

Resumable: an image whose `index.json` already exists is skipped unless `--overwrite`.
Roughly 35 s per image on this box's shared GPU (~25 min for all 40).

**Size.** One JSON per patch, so the browser only ever fetches the patch you clicked
(~45 KB). Whole payload is ~470 MB — COCO ~9 MB/image, counting ~15 MB/image (counting is
larger because every patch carries all 51 number variants, while most COCO patches carry
none). Token strings are pooled into a per-image `vocab` table and referenced by index;
probabilities are rounded to 4 dp and centered logits to 1 dp, both finer than displayed.
Storing every word column at every patch instead made a single 192-patch image 22 MB, so
each patch stores only the columns it can actually display.

## Layout

```
UI/
  README.md          (this file)
  build_payload.py   GPU builder
  index.html         the UI (no dependencies, no build step)
  images/<dataset>/  the images themselves, copied so the site is self-contained
  data/
    index.json                     datasets -> image lists
    <dataset>/<stem>/index.json    grid, groups+words, patch->group map, prompt, vocab
    <dataset>/<stem>/patches/N.json  one patch: top-10 + centered logits, per method/layer
```
