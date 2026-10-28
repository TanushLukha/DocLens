# DocLens: reading printed Hindi and English documents

DocLens reads printed Hindi and English documents: it finds the text on a page, reads it, and
pulls out the fields that matter, such as the items and the total on a receipt. Every part is
built from first principles in Python, OpenCV and PyTorch, and every part is measured against
Tesseract and modern open OCR models on the same test data.

Why Hindi and English together? Indian paperwork mixes Devanagari and Latin script on one page,
often in one line. Devanagari is also hard to read by machine: letters join under a headline
(the shirorekha), consonants fuse into conjuncts and vowel signs sit before, above or below their
consonant, so a word cannot be cut into separate characters the way an English word can.

## Roadmap

| Status | Milestone | What gets built |
|---|---|---|
| code ready | **M1 Baseline** | test sets (real Hindi words, real receipt words and lines, synthetic lines under eight kinds of damage); CER and WER; Tesseract measured three ways; OpenCV image cleaning compared |
| planned | **M2 Recognizer** | a CRNN trained with CTC loss from scratch on synthetic lines and real words; greedy and beam decoding; rescoring with TinyGPT; tested on fonts it never saw |
| planned | **M3 Pages** | finding lines and blocks (classical against learned), reading order, whole page accuracy |
| planned | **M4 Fields** | key information extraction on receipts: layout aware transformers (LayoutLMv3, LiLT) against a text only model, what layout adds, explanations |
| planned | **M5 Comparison** | DocLens against PaddleOCR, Surya, EasyOCR and a small vision language model: accuracy, speed, memory, licence |
| planned | **M6 Ship** | FastAPI service, Docker image, ONNX export, a demo that runs in the browser |

The plan, the data and every design decision (what, why, and why not the alternatives) are in
[docs/PLAN.md](docs/PLAN.md).

## Run M1 on a Mac

Once per machine: Tesseract, and FriBiDi so that Pillow can shape Devanagari
(why the second line is needed: see "If something fails" below).

```bash
brew install tesseract fribidi
echo 'export DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib:/usr/local/lib:/usr/lib' >> ~/.zshrc
source ~/.zshrc
```

The environment, then the checks:

```bash
cd ~/"Tanush's Code/DocLens"
python3.13 -m venv .venv && source .venv/bin/activate
pip install -e ".[data,dev]"
python scripts/check_env.py
```

The study:

```bash
python scripts/get_data.py                 # Mozhi test and val, CORD test, Tesseract models, English text
pytest -q                                  # 27 tests
python scripts/make_synth.py               # the synthetic test sets, from TinyGPT's held out text
python scripts/eval_tesseract.py --quick   # 200 images per set: checks the whole pipeline
python scripts/eval_tesseract.py           # the full study; interrupted runs continue where they stopped
```

Or all of it in one go: `bash scripts/run_m1.sh`.

Results: [docs/results/m1_tesseract.md](docs/results/m1_tesseract.md) (tables) and
`docs/results/m1_tesseract.json` (every number). Predictions are kept in `runs/m1/`.

## What M1 measures

* **Test sets.** Mozhi Hindi: 10,173 printed words cropped from book scans at 600 DPI, each
  transcribed by hand. CORD v2: 100 photographed receipts, read as single words and as whole lines.
  Synthetic: up to 750 held out lines (Hindi, English and mixed), drawn once with the training fonts
  and once with six fonts kept out of all training, each line clean and under seven kinds of damage.
* **Metrics.** Character and word error rates on Unicode normalised text, the median per image,
  the share of images that fail badly (CER above 50%) and the share read perfectly.
* **Experiments.** How much background a crop needs, and which page mode copes with it;
  Tesseract's fast models against its best models; `-l hin`, `-l eng` and `-l hin+eng` on every
  set, counting outputs in the wrong script; eight OpenCV cleaning pipelines on every set; the
  characters Tesseract confuses most in Hindi. Each set is read in its own language: Hindi for
  Mozhi, English for CORD, both for the synthetic lines.

## Layout

```
src/doclens/
  text.py         Unicode normalisation, script detection
  metrics.py      edit distance, alignment, CER and WER
  fonts.py        the bundled fonts, their scripts, splits and licences
  render.py       drawing a line of Hindi, English or both, shaped correctly
  degrade.py      blur, noise, low resolution, JPEG, skew, shadow, phone photo
  preprocess.py   grey, scaling, denoising, CLAHE, Otsu, Sauvola, deskewing
  tesseract.py    batched, parallel Tesseract
  datasets.py     Mozhi, CORD and the synthetic sets, read into one shape
  corpus.py       held out text lines from TinyGPT's corpus
scripts/          check_env, get_data, make_synth, eval_tesseract, run_m1.sh
tests/            unit tests for all of the above
fonts/            35 font files with their licences (fonts/README.md)
docs/             PLAN.md and results/
data/, runs/      downloads and outputs, never committed
```

## Data and licences

| What | Source | Licence |
|---|---|---|
| Mozhi Hindi printed words | [IIIT Hyderabad, NLTM OCR](https://ilocr.iiit.ac.in/dataset/7/) | CC BY 4.0 |
| CORD v2 receipts | [Clova AI](https://github.com/clovaai/cord), via the Hugging Face Hub | CC BY 4.0 |
| Tesseract models (fast and best) | [tessdata_fast](https://github.com/tesseract-ocr/tessdata_fast), [tessdata_best](https://github.com/tesseract-ocr/tessdata_best) | Apache 2.0 |
| English sentences | `findnitai/english-to-hinglish` on the Hugging Face Hub | Apache 2.0 |
| Hindi and Hinglish text | TinyGPT's held out split (Hindi Wikipedia and Hinglish sources) | as in TinyGPT |
| Fonts | see [fonts/README.md](fonts/README.md) | OFL 1.1, Apache 2.0, GPL with font exception, and others |

`get_data.py` records the SHA256 of every downloaded file in `data/MANIFEST.json`, so each result
can name the exact data and models it came from.

## If something fails

* **`check_env.py` says Pillow cannot shape Devanagari.** Pillow's own wheels include libraqm, which
  draws Devanagari correctly, but it loads the FriBiDi library when Python starts, looking for it by
  name and in `/usr/local/lib`. Homebrew on Apple silicon keeps it in `/opt/homebrew/lib`, which is
  not searched unless `DYLD_FALLBACK_LIBRARY_PATH` says so: run the three setup lines above, open a
  new terminal, activate the environment and check again. Without shaping, DocLens refuses to draw
  Hindi rather than draw it wrong.
* **`tesseract not found`.** `brew install tesseract`.
* **A `SKIP` line in `eval_tesseract.py`.** That set's data is missing: run `get_data.py` again.

## Licence

The code is MIT licensed ([LICENSE](LICENSE)). Fonts, datasets and models keep their own licences,
listed above.
