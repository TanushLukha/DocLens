# DocLens plan

## What DocLens is, and why it is worth building

DocLens reads printed documents in Hindi and English: it finds the text on a page, reads it, and
extracts the fields that matter (on a receipt: items, prices, total). It is built in six measured
steps, and each step is compared with the tools people actually use: Tesseract, PaddleOCR, Surya,
EasyOCR and a small vision language model.

What it shows, in one project:

* **Computer vision with OpenCV:** cleaning images (denoising, contrast, binarisation, deskewing)
  and finding text lines and blocks.
* **Deep learning from scratch:** a CRNN recognizer trained with CTC loss in PyTorch, with its
  own data pipeline, decoding and language model rescoring (using TinyGPT, project 1).
* **Indic text done properly:** Unicode normalisation, Devanagari shaping, error analysis at the
  level of vowel signs and conjuncts.
* **Document AI with Hugging Face Transformers:** layout aware models (LayoutLMv3, LiLT) for key
  information extraction, with an ablation of what layout adds and explanations of the
  predictions.
* **Evaluation discipline:** fixed test sets, held out fonts, baselines first, numbers that trace
  back to files.
* **Deployment:** FastAPI, Docker, ONNX, a demo that runs in the browser.

The question running through it: on printed Hindi and English, how far is a small model built
from scratch from Tesseract and from today's large open models, in accuracy, speed, memory and
licence, and where does each one fail?

## Milestones

| Milestone | Question | Built | Measured on |
|---|---|---|---|
| M1 Baseline | How well does Tesseract read Hindi, English and mixed lines, and what does image cleaning buy? | test sets, metrics, batched Tesseract, eight cleaning pipelines | Mozhi test words, CORD test words and lines, synthetic lines |
| M2 Recognizer | Can a small CRNN trained from scratch beat Tesseract, also on fonts it never saw? | synthetic training lines drawn on the fly, CRNN with CTC, greedy and beam decoding, TinyGPT rescoring | the M1 sets, same metrics |
| M3 Pages | How well are lines and blocks found, and does reading order hold on whole pages? | classical line finding (projections, connected components) against a learned detector; reading order | page CER, detection precision and recall |
| M4 Fields | What does layout add to extracting fields? | LayoutLMv3 and LiLT against a text only model; true text against OCR text; explanations | CORD field F1 |
| M5 Comparison | Where does DocLens stand against modern OCR? | one harness for PaddleOCR, Surya, EasyOCR, a small VLM and DocLens | the M1 sets: accuracy, images per second, memory, licence |
| M6 Ship | Can someone use it? | FastAPI service, Docker image, ONNX export, browser demo | latency and throughput, an end to end test |

## Data

| Set | What | Size | Licence | Used in |
|---|---|---|---|---|
| Mozhi Hindi | printed Hindi words cropped from 1,000 book pages scanned at 600 DPI, transcribed by hand | 79,762 train, 10,114 val, 10,173 test | CC BY 4.0 | M1 test; M2 train and validation |
| CORD v2 | photographed Indonesian receipts (Latin script) with the box and text of every word and the field each line belongs to | 800 train, 100 validation, 100 test | CC BY 4.0 | M1 test; M2 train and validation; M4 |
| Synthetic lines | held out sentences from TinyGPT (Hindi, Hinglish) and English sentences, drawn with known fonts and damaged in known ways | up to 750 lines, each under 8 conditions, with seen and unseen fonts | ours | M1, M2 test |
| Synthetic validation lines | held out sentences not in the test sets, training fonts, one random condition each | 901 lines | ours | M2 validation |
| Training text | lines from TinyGPT's training documents and English training sentences | 486,962 lines | as their sources | M2 train |
| Tesseract models | fast and best LSTM models for Hindi, English, script detection | 6 files | Apache 2.0 | M1, M5 |

## How results are scored

* **CER** (character error rate): edits (insertions, deletions, substitutions) needed to turn the
  output into the true text, divided by the length of the true text, counted in Unicode code
  points after NFC normalisation. The rate of a set is total edits over total characters, so long
  lines weigh more. **WER** is the same over words.
* **Median CER** and **catastrophic share** (images with CER above 50%): an average can look fine
  while a fifth of the images are unreadable; these two show it.
* **Exact**: the share of images read perfectly.

## Decisions in M1

Each decision: what we did, why, and why not the alternatives.

**1. Measure before building.**
What: before any model, Tesseract is measured on fixed test sets with fixed metrics.
Why: every later claim ("the CRNN reaches 3% CER") only means something against the same yardstick.
Why not start with a model: without a baseline we could not tell whether a gain came from the
model or from easier test data.

**2. Three kinds of test data.**
What: real Hindi print (Mozhi), real photos of Latin script receipts (CORD), and synthetic lines.
Why: the real sets keep the numbers honest; the synthetic set changes one thing at a time (one
kind of damage, seen or unseen fonts), has exact ground truth, and contains mixed Hindi and
English lines, which neither real set covers.
Why not only real data: a real scan mixes several problems at once, so it cannot say which one
hurts. Why not only synthetic: a model can learn the quirks of our renderer; the real sets catch that.

**3. Synthetic text from TinyGPT's held out split.**
What: Hindi and Hinglish sentences from TinyGPT's `val.jsonl`, English sentences from the
findnitai corpus (0.5% held out by a hash of the sentence), packed into lines of 25 to 70 characters.
Why: real sentences carry real word and conjunct frequencies; the held out split is never used
for training, here or in TinyGPT, so rescoring with TinyGPT in M2 cannot have memorised a test line.
Why not random characters: real Hindi follows patterns that random strings never show, so a
recognizer would be tested on text it will never meet. Why not new text: TinyGPT's corpus is
already cleaned and deduplicated.

**4. Fonts in the repository, split by font.**
What: 35 font files with their licences; 29 for training and the "seen" test, 6 held out
(Annapurna SIL regular and bold, Gargi, Caladea, Lato regular and bold).
Why: the same pixels on every machine, and a fair test of generalisation: the "unseen" set uses
letter shapes no model here is ever trained on.
Why not system fonts: they differ between macOS and Linux, so results would not reproduce.

**5. Shaping is required.**
What: text is drawn through libraqm (HarfBuzz); DocLens refuses to draw without it, and checks that
every Devanagari font joins the conjunct क्ष.
Why: without shaping, conjuncts and vowel signs come out as loose glyphs: wrong Hindi that still
looks like Hindi, which would poison both training and testing.
Why not Pillow's basic layout: that is exactly the broken output.

**6. Damage that is named, seeded and isolated.**
What: eight conditions (clean, blur, noise, low resolution, JPEG, skew, shadow, phone photo); every
line is saved under all eight with the same text and font.
Why: conditions compare like for like, and the damage ranges follow a published Devanagari stress
test (blur sigma 1.0 to 1.8, noise on 6% of pixels, resolution scaled to 0.45).
Why not random mixed augmentation: errors could not be traced to a cause.

**7. CER on NFC code points, plus median and catastrophic share.**
What: as in "How results are scored", with zero width joiners kept.
Why NFC: the same visible letter can be stored two ways (a nukta letter as one code point or as
base plus nukta); NFC makes them equal before counting. Why median and catastrophic share: the
stress test recommends them, because a mean hides how many images fail outright. Why not akshara level rates first: code points are
what OCR engines output and the usual unit in published numbers; an akshara rate can be added
later from the same alignment.

**8. Tesseract's models and engine.**
What: LSTM engine (`--oem 1`), `--dpi 300`; fast and best models downloaded into `data/`, with their
SHA256 in `data/MANIFEST.json`.
Why: Homebrew's Tesseract comes with English and script detection only, from the 4.1.0 fast set;
downloading both sets ourselves makes "fast against best" an explicit comparison and gives every
machine identical models. The resolution is given because the images handed to Tesseract carry
none; one value for every set, so that differences come from the data.
Why not the legacy engine: the fast and best model sets contain LSTM models only. Why not the
`tesseract-lang` package: over a hundred languages of fast models, none of them best.

**9. Tesseract run in batches, in parallel.**
What: images are passed 100 at a time through a list file and the output is split on form feeds;
one single threaded process per core.
Why: starting Tesseract and loading its models takes over a tenth of a second, longer than reading
a word, so one process per image wastes most of the time; separate processes scale with cores.
Why not pytesseract: it starts one process per image. Why not tesserocr: it must be compiled against
the installed Tesseract, one more thing to break on every machine.

**10. Image cleaning is measured, not assumed.**
What: eight pipelines (raw; scaled to 64 pixels high; plus median denoising, CLAHE, Otsu, Sauvola
or deskewing; and all of them together) on every set.
Why: advice like "binarise first" comes from older OCR engines; whether a step helps a modern
recognizer depends on the damage, so it is measured per condition.
Why not choose by eye: an image can look cleaner and read worse.

**11. Every prediction is cached.**
What: one file per set, models and pipeline in `runs/m1/`.
Why: a long run can stop and continue, and tables can be rebuilt without running OCR again.
Why not keep results only in memory: one crash near the end would lose the whole run.

**12. Page mode 7 for every crop, and a margin around Mozhi words.**
What: every crop is read as one line of text (`--psm 7`); Mozhi crops get a border of paper colour
(the image's 90th percentile) a quarter of their height wide. The margins experiment measures this
on Mozhi validation words, never on the test words.
Why: the first full run (page mode 8, no border, `-l hin+eng`) got a CER of 26% on Mozhi. In checks
on samples of the test words, a border alone roughly halved that, and reading in Hindi only halved
it again (decision 13). Mozhi crops end at the ink. Page modes 8 and 13 react to the border both
ways: too little and too much both hurt (they seem to treat the whole image as the text, so a wide
border shrinks it). Page mode 7 finds the text inside the crop and stays flat once there is
enough background.
On 2,000 validation words: no border 21.6%, a quarter of the height 6.2% (page mode 7), and page
mode 8 is best at a tenth (5.8%) but reaches 60.2% at a full height.
Why not page mode 8, the mode Tesseract names for single words: it is right only for one border
size. Why not a border on every set: CORD crops already keep real background, and in a check on
the CORD test words a white frame around the grey receipt raised the CER from 15% to 70%.

**13. Each set read in its own language, and the cost of mixing them measured.**
What: `-l hin` for Mozhi, `-l eng` for CORD, `-l hin+eng` for the synthetic lines, which mix both;
the languages experiment runs all three settings on every set and counts outputs that contain
letters of the wrong script.
Why: on a short crop, `hin+eng` has no context to choose a script, and the first run read Hindi
words as Latin letters ("कमरे" as "FAR") and receipt words as Devanagari ("CP" as "पक"). A careful user who knows the language sets it, and the baseline should be Tesseract
at its best.
Why not `hin+eng` everywhere: that measures the mix up, not reading. Why not detect the script
of each crop first: that is part of what DocLens must do itself (M2, M3), not of the baseline.

**14. FriBiDi on the Mac through `DYLD_FALLBACK_LIBRARY_PATH`.**
What: `brew install fribidi` plus one line in `~/.zshrc`.
Why: Pillow's wheels include libraqm but load FriBiDi when Python starts, by name and from
`/usr/local/lib` (read in Pillow's source, `src/thirdparty/fribidi-shim/fribidi.c`); Homebrew on
Apple silicon keeps it in `/opt/homebrew/lib`, which macOS searches only when this variable says so.
Why not build Pillow from source: it works too, but needs a compiler and Homebrew's image
libraries, and the next `pip install --upgrade` silently brings back the wheel.

## What M1 found, and what it means for M2

The full tables are in `docs/results/m1_tesseract.md`, the headline numbers in the README.

* **The yardstick.** Tesseract's best models, each set in its own language: Mozhi 6.5% CER,
  synthetic lines 4.2% (seen fonts) and 3.2% (unseen fonts), CORD words 19.9%, CORD lines 42.8%.
  M2 is scored on the same images with the same code.
* **One model for both scripts.** Tesseract must be told the language, and telling it both costs
  accuracy on single script text: 3.2 points of CER on Mozhi, 9.9 on CORD words. M2 learns one alphabet holding both scripts and is
  never told which one a line uses.
* **Train with damage instead of cleaning.** No fixed pipeline helps on average, and each filter
  helps only its own damage. M2 draws its training lines through the eight damage conditions at
  random, so robustness is learned rather than bolted on.
* **Crops need background, in training as in testing.** Training crops get the same margin as test
  crops (a quarter of the height for Mozhi, padding for synthetic lines), so the model never meets
  ink touching the edge only at test time.
* **Punctuation counts.** A lone danda is most of Tesseract's outright failures on Mozhi, so M2's
  alphabet keeps । and ॥ as characters of their own, and the scorer keeps counting them.
* **Results depend on the machine.** The same Tesseract models left 29% of CORD lines empty on the
  Mac (5.5.3) and 11% in a Linux container (5.3.4). Every comparison runs on the M2 MacBook, and
  each results file names the versions it used.

## Decisions in M2

**15. A CRNN trained with CTC, one model for both scripts.**
What: seven convolutions with batch normalisation turn every 4 pixel wide slice of a line into a
feature vector; two bidirectional LSTM layers (256 units each way) read the slices in both
directions; a linear layer scores 258 classes per slice. About 4 million parameters. CTC loss.
Why: the standard recognizer for printed lines, small enough to train on a laptop GPU, and CTC
needs no character positions, only the text of each line. One alphabet holds Devanagari and
Latin, so the model is never told the language (M1: telling Tesseract both costs accuracy).
Why not an attention or transformer decoder (TrOCR style): it needs far more data and compute,
and on long lines it can skip or repeat words; CTC's left to right alignment fits printed text.
Why not fine tune a pretrained model: M2's question is what a model built from scratch reaches;
pretrained readers come in M5.

**16. Code points as output units.**
What: each class is one Unicode code point after NFC (257 characters plus the blank); a conjunct
such as क्ष is read as क, virama, ष.
Why: the scorer counts code points, any conjunct can be written, and the output layer stays
small. Why not aksharas (written syllables): thousands of classes, most of them rare, and a
conjunct missing from training could never be read.

**17. Training text and alphabet from training data only.**
What: lines from TinyGPT's training documents and English training sentences (300,000 Hindi,
150,000 Latin, all 36,962 mixed lines), Mozhi's training words and CORD's training receipts. The
alphabet is every character of the real data plus the synthetic characters seen 50 times or more
that a training font can draw.
Why: nothing from a validation or test set shapes the model; a test character outside the
alphabet stays an error, and the results report how often that happens.

**18. Three sources: synthetic lines 50%, Mozhi words 30%, CORD crops 20%.**
What: every training sample comes from one of the three by these shares; CORD's 800 training
receipts give its training word and line crops.
Why: synthetic lines bring unlimited Hindi, English and mixed text with exact labels and named
damage; Mozhi brings real Hindi print; CORD brings real receipts (thermal print, folds, phone
photos), a look no font of ours has. Why not synthetic only: a second run (`--synthetic-only`)
measures exactly what the real data adds. Why use CORD's training split in M2 and not only in M4:
M4 extracts fields from OCR text, so it needs a recognizer that reads receipts.

**19. Damage drawn at random at every step.**
What: 30% of synthetic lines stay clean; the rest get one of M1's seven damage conditions, and 20%
of those a second one; real crops are damaged 30% of the time. Sizes 20 to 44 pixels, ink and
paper colours and margins vary; a third of the lines are cut to one to four words, some Latin
lines are capitalised, 5% are prices and quantities in receipt styles.
Why: M1 found that no fixed cleaning helps on average; a model that has seen the damage learns
to read through it. Why not a fixed, pre rendered training set: every pass would repeat the same
pixels; drawing fresh costs CPU time, which the data workers spend while the GPU trains.

**20. Images 48 pixels high, aspect kept, contrast stretched.**
What: each crop is scaled to 48 pixels high (lines up to 32 times as wide), then its 2nd
percentile (ink) maps to minus 1 and its 98th (paper) to plus 1.
Why: Devanagari puts marks above and below the headline; Mozhi's crops are 48 pixels high, so they
keep their resolution. The stretch puts faded thermal print and dark shadowed scans on one scale.
Why not a fixed width: it would squash long lines and stretch short words. Why not binarise
first: M1 showed global thresholding hurting under shadow. The benchmark (decision 24) times
heights 32 and 48 before the long run.

**21. Batches of similar width, and an LSTM that never reads padding.**
What: each data worker draws 8 batches' worth of samples, sorts them by width and cuts them into
batches; batch widths are rounded up to a multiple of 64 pixels. In each LSTM layer the left to
right pass reads the padded batch as it is, and the right to left pass reads every line reversed
within its own length, so for both the padding comes after the last real column.
Why: a batch costs as much as its widest image, so mixing words with long lines would spend most
of the compute on padding; the Apple GPU builds its compute graph once per tensor shape, so a few
dozen widths run fast where a new width at every step would not. The first trial on the Mac used
PyTorch's packed sequences for the LSTM and trained at 18 samples per second, an eighth of the
benchmark's estimate: on 64 lines of 100 to 300 columns the packed LSTM takes 5.9 times as long as
this one even on a CPU (`scripts/bench_m2.py` times both on the GPU). Why not let the LSTM run over
the padding: a line next to a wider one would then read differently, since the right to left pass
would cross the neighbour's padding first. A test checks that the two directions match PyTorch's
own bidirectional LSTM exactly, and that padding changes no real column.

**22. Optimisation.**
What: AdamW at a learning rate of 0.001 with weight decay 0.0001, 1,000 warm up steps, cosine decay to 0.00001, gradients
clipped at 5, batches of 64, CTC computed on the CPU. Length and precision are set from the
benchmark's numbers. A checkpoint every 1,000 steps; `--resume` continues from the last one.
Why: the usual recipe for CRNNs; clipping guards the LSTM against rare large gradients. CTC on the
CPU because its kernel exists on every PyTorch build, and it is a small part of a step (the
benchmark checks the Apple GPU kernel and times both).

**23. Model selection on validation data only.**
What: every 1,000 steps the model reads 2,000 Mozhi validation words, 901 synthetic validation
lines and CORD's validation words and lines; the checkpoint with the lowest mean CER over these
sets is kept. The synthetic validation lines are held out text that is not in the test sets,
drawn with training fonts only (the held out fonts are never used to choose anything).
Why: test sets are read once, at the end. Only 124 held out lines mix both scripts and the test
sets took 117 of them, so mixed lines are measured on the test sets only.

**24. A benchmark before the long run.**
What: `scripts/bench_m2.py` checks that every piece of a step runs on the GPU and times training
at heights 32 and 48, with and without mixed precision, plus the data workers' speed.
Why: the run takes hours on a laptop; ten minutes of measurement choose its settings.

**25. Decoding: greedy, beam search, then TinyGPT.**
What: greedy decoding; prefix beam search with 10 candidates; the candidates rescored by
log P_ctc + lam log P_TinyGPT + beta length, with lam and beta chosen on the validation sets,
once for lines holding Devanagari and once for the rest.
Why: rescoring lets a language model fix what the pixels leave open (a vowel sign, a look alike
consonant) without being built into the search; TinyGPT reads BPE tokens while the beam grows
characters, so scoring finished candidates is the clean way to join them. Its weight is tuned
because it read Hindi and Hinglish, not receipts. Why not an n gram model: TinyGPT beat them in
its own M2 and M3.

**26. Intervals and paired tests.**
What: every CER comes with a bootstrap 95% interval (2,000 resamples of images); the CRNN and
Tesseract are compared on the same images with a paired bootstrap and a p value.
Why: a test set is a sample; an interval says how far the number could move on another sample,
and pairing removes the part of that spread that comes from which images are hard.

**27. Every push checked, every run recorded.**
What: GitHub Actions runs ruff, mypy and all tests on Linux with the CPU build of PyTorch; tests
that need downloaded data skip there. Training runs go to MLflow (SQLite in `runs/mlflow.db`) as
well as to `metrics.jsonl`.
Why: a change that breaks a test is caught before it reaches a long run. MLflow's own
documentation names SQLite its default store and its old file store "to be deprecated soon".
Why not Weights and Biases: an account and a network service for what fits in one local file.

## Decisions ahead

Recorded here when each milestone is built. Already known:

* **M4 licences.** LayoutLMv3's model page lists CC BY NC SA 4.0 (non commercial) while its README
  says MIT; LiLT is MIT. LiLT is the model to ship, LayoutLMv3 is for comparison only.
* **M5 licences.** Surya's code is Apache 2.0 but its weights use a modified OpenRAIL M licence
  with limits on commercial use; PaddleOCR is Apache 2.0. The comparison reports licence next to
  accuracy, because it decides what can be shipped.

## What runs where

Everything that produces a number runs on the M2 MacBook. M1 needs only the CPU (Tesseract uses
every core); M2 trains on the Apple GPU through PyTorch's MPS backend, with data workers on the
CPU cores. GitHub Actions runs the tests on Linux after every push.
