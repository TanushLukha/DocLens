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
| M3 Pages | How well are lines and blocks found, and does reading order hold on whole pages? | classical line finding (projections, connected components, XY cut) against docTR's pretrained detector and Tesseract's page analysis; reading order | synthetic pages and CORD receipts: page CER, line precision and recall, reading order |
| M4 Fields | What does layout add to extracting fields? | LayoutLMv3 and LiLT against a text only model; true text against OCR text; explanations | CORD field F1 |
| M5 Comparison | Where does DocLens stand against modern OCR? | one harness for Tesseract, PaddleOCR, EasyOCR, two small vision language models (PaddleOCR VL, Surya 2) and DocLens | samples of the M1 sets, M3's synthetic pages and CORD's receipts: CER, page CER, field F1 through M4's extractors, speed, memory, licence |
| M6 Ship | Can someone use it? | FastAPI service, Docker image, ONNX export, browser demo | latency and throughput, an end to end test |

## Data

| Set | What | Size | Licence | Used in |
|---|---|---|---|---|
| Mozhi Hindi | printed Hindi words cropped from 1,000 book pages scanned at 600 DPI, transcribed by hand | 79,762 train, 10,114 val, 10,173 test | CC BY 4.0 | M1 test; M2 train and validation |
| CORD v2 | photographed Indonesian receipts (Latin script) with the box and text of every word and the field each line belongs to | 800 train, 100 validation, 100 test | CC BY 4.0 | M1 test; M2 train and validation; M4 |
| Synthetic lines | held out sentences from TinyGPT (Hindi, Hinglish) and English sentences, drawn with known fonts and damaged in known ways | up to 750 lines, each under 8 conditions, with seen and unseen fonts | ours | M1, M2 test |
| Synthetic validation lines | held out sentences not in the test sets, training fonts, one random condition each | 901 lines | ours | M2 validation |
| Synthetic pages | held out Hindi and English text set as A4 pages at 300 dpi, one and two columns, with every line's box, text and order | 300 test pages (150 per font set), 150 validation pages | ours | M3 |
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
directions; a linear layer scores 260 classes per slice. About 4 million parameters. CTC loss.
Why: the standard recognizer for printed lines, small enough to train on a laptop GPU, and CTC
needs no character positions, only the text of each line. One alphabet holds Devanagari and
Latin, so the model is never told the language (M1: telling Tesseract both costs accuracy).
Why not an attention or transformer decoder (TrOCR style): it needs far more data and compute,
and on long lines it can skip or repeat words; CTC's left to right alignment fits printed text.
Why not fine tune a pretrained model: M2's question is what a model built from scratch reaches;
pretrained readers come in M5.

**16. Code points as output units.**
What: each class is one Unicode code point after NFC (259 characters plus the blank); a conjunct
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
measures exactly what the real data adds; its checkpoint is chosen on the same validation sets as
the main run's, so the two differ in training data only. Why use CORD's training split in M2 and not only in M4:
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
What: each crop is scaled to 48 pixels high (lines up to 24 times as wide), then its 2nd
percentile (ink) maps to minus 1 and its 98th (paper) to plus 1.
Why: Devanagari puts marks above and below the headline; Mozhi's crops are 48 pixels high, so they
keep their resolution. The stretch puts faded thermal print and dark shadowed scans on one scale.
Why not a fixed width: it would squash long lines and stretch short words. Why not binarise
first: M1 showed global thresholding hurting under shadow. The benchmark (decision 24) times
heights 32 and 48 before the long run.

**21. Five fixed batch shapes, and an LSTM that never reads padding.**
What: every image goes to the narrowest of five width classes that holds it, 128, 256, 512, 768
and 1,152 pixels at height 48, and is padded to that width. A training batch holds a fixed number
of images for its class, as many as fit 32,768 pixel columns, at most 256: 256, 128, 64, 42 and
28. Each data worker keeps a bin per class and sends a batch when a bin is full. Reading uses the
same five widths in batches of 64, the last one filled with blank images, and TinyGPT scores
candidates in batches of a fixed number of rows, with lengths rounded up to 16 tokens. Lines are
at most 24 times as wide as they are high. In each LSTM layer the left to right pass reads the
padded batch as it is, and the right to left pass reads every line reversed within its own length,
so for both the padding comes after the last real column.
Why: PyTorch's Apple GPU backend compiles and keeps a graph for every tensor shape it meets. The
first full run cut batches from sorted pools by the pixel budget alone, so they came in many sizes
and widths: a simulation of that batching with the run's settings gives 83 shapes in the first 250
steps, and a mean batch of 86 images, as the run's log showed. GPU memory reached 19.3 GB by step
100 with 25 GB of swap in use, the run crawled at 58 to 73 samples per second, and before step 300
it stopped with "MPS backend out of memory": 2.45 GiB in PyTorch's tensors, 17.52 GiB in other
allocations. Five classes leave ten shapes in all, five for training and five for reading, and the
budget still gives every training step about the same memory and time. Padding barely changes:
28.0% of the pixel columns, against 26.5% for batches cut from sorted pools (the same simulation,
8,192 samples). On the Mac, `scripts/bench_m2.py` measured both: 100 steps on the five shapes
held GPU memory at 5.3 GB (reading one batch of each width, as validation does, took it to 6.9
GB), and 16 more steps on new shapes took it to 9.8 GB. The full run then stayed between 5.3 and
5.6 GB for all 20,000 steps, at a median 216 samples per second against 58 to 73 before.
Why not more classes: each adds two shapes to build and keep. Why not fewer: they pad more; with
one width of 1,152 pixels, nine tenths of a batch of words would be padding. Why not a fixed
number of images at any width: the run before, with batches of 64, crawled at 18 to 30 samples per
second, and one step on 64 of the widest lines needs 5.5 GB where 64 typical lines need 1.6 GB.
The LSTM: the first trial used PyTorch's packed sequences, which on the Mac took 4.7 times as long
as this LSTM (1,609 against 343 ms for 64 lines of 100 to 300 columns, forward and backward). Why
not let the LSTM run over the padding: a line next to a wider one would read differently, since the
right to left pass would cross the neighbour's padding first. Tests check that the two directions
match PyTorch's own bidirectional LSTM exactly, that padding changes no real column, and that
reading in batches filled with blanks gives the same scores as reading each image alone.

**22. Optimisation.**
What: AdamW at a learning rate of 0.001 with weight decay 0.0001, 1,000 warm up steps, cosine decay
to 0.00001, gradients clipped at 5, 20,000 steps, full precision, CTC on the CPU, four data
workers. Each image's loss is divided by its text length; the batch's losses are
summed and divided by 64, so every image weighs the same whether it came in a batch of 28 lines or
of 256 words. A checkpoint every 1,000 steps; `--resume` continues from the last one.
Why: the usual recipe for CRNNs; clipping guards the LSTM against rare large gradients. Mixed
precision bought nothing on the Mac's GPU (fp16 406 against 396 ms per step of lines, bf16 448), and
CTC runs as fast on the CPU as on the GPU (396 against 393 ms), where its kernel exists on every
PyTorch build. Four workers draw over 1,400 synthetic lines per second (361 each in the last
benchmark), several times what the GPU uses (about 215 samples per second in the main run, whose
log shows the data wait at 0% after the first 100 steps), and leave memory and cores to the
training process.

**23. Model selection on validation data only.**
What: every 1,000 steps the model reads 2,000 Mozhi validation words, 901 synthetic validation
lines and CORD's validation words and lines; the checkpoint with the lowest mean CER over these
sets is kept. The synthetic validation lines are held out text that is not in the test sets,
drawn with training fonts only (the held out fonts are never used to choose anything).
Why: test sets are read once, at the end. Only 124 held out lines mix both scripts and the test
sets took 117 of them, so mixed lines are measured on the test sets only.

**24. A benchmark before the long run, and a log that shows the machine.**
What: `scripts/bench_m2.py` checks that every piece of a step runs on the GPU, compares the two
LSTMs, times training steps on batches of words, lines and the longest lines at heights 32 and 48
with the GPU memory each needs, measures the data workers' speed, then follows GPU memory over 100
steps on the five batch shapes and over 16 steps on new shapes. The training log prints samples
per second, the average batch, GPU memory (all of it, and the part in tensors) and swap every 100
steps.
Why: the run takes hours on a laptop; minutes of measurement choose its settings. The first
benchmarks timed one batch shape at a time, so they could not show memory growing with the number
of shapes; the full run showed it, and now the benchmark does too.

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
Two training runs are compared the same way (`scripts/compare_runs.py`), from each run's cached
readings.
Why: a test set is a sample; an interval says how far the number could move on another sample,
and pairing removes the part of that spread that comes from which images are hard. Why not
compare two intervals side by side: much of each one's width is which images happen to be hard,
which pairing cancels. On the unseen fonts, the paired interval of the two runs' difference is
0.08 points wide, narrower than either run's own interval (0.16 and 0.12).

**27. Every push checked, every run recorded.**
What: GitHub Actions runs ruff, mypy and all tests on Linux with the CPU build of PyTorch; tests
that need downloaded data skip there. Training runs go to MLflow (SQLite in `runs/mlflow.db`) as
well as to `metrics.jsonl`.
Why: a change that breaks a test is caught before it reaches a long run. MLflow's own
documentation names SQLite its default store and its old file store "to be deprecated soon".
Why not Weights and Biases: an account and a network service for what fits in one local file.

## What M2 found, and what it means for M3

The full tables are in `docs/results/m2_crnn.md`, `m2_crnn_synthetic.md` and
`m2_main_vs_synthetic.md`, the headline numbers in the README.

* **The recognizer is ready for pages.** One CRNN reads every M1 test set better than Tesseract
  without being told the language: Mozhi 0.6% against 6.5%, synthetic lines 0.7% and 0.8% (seen
  and unseen fonts) against 4.2% and 3.2%, CORD words 1.3% against 19.9%, CORD lines 2.7% against
  42.8%, every difference with p < 0.001. Those crops were cut from true boxes; M3 feeds it lines
  found on whole pages, so page CER adds finding errors to reading errors, and M3 reports both.
* **Skew is the one damage left.** Lines turned by 2 to 5 degrees hold about half of the CRNN's
  errors on the synthetic lines (2.7% and 3.0% CER, against at most 0.6% under any other damage).
  A line finder knows each line's angle, so M3 straightens every line before reading it and
  measures that against training with more skew. M1's other cleaning filters are not needed in
  front of the CRNN.
* **Real images of the target kind matter most.** Trained on synthetic lines only, the same model
  still beats Tesseract on every set, but real Mozhi and CORD training images cut its error on
  those sets by 84% to 92%, for 0.2 points more on synthetic lines. Synthetic lines alone leave
  3.9% on Mozhi and 16.8% on CORD words, so for a new source of pages, real training crops from it
  are the first thing to look for.
* **Comparisons must say what each reader trained on.** The main model trained on Mozhi's and
  CORD's training splits; Tesseract did not, and the M5 models may not have. The synthetic only run
  and the unseen fonts are the fair out of the box comparisons, so M5 reports both runs.
* **Keep TinyGPT rescoring, tuned per model.** It never made a set measurably worse, helped on
  unseen fonts, and helped most where the model was unsure (the synthetic only run: Mozhi 4.6% to
  3.9%, CORD words 19.0% to 16.8%). Its weights are chosen again for every model: the synthetic
  only model wanted 0.6 where the main model wanted 0.3 to 0.4.
* **Training crops must be cut like test crops.** A lone danda, Tesseract's commonest failure in
  M1, is also what the synthetic only model never learned (all 576 misread): its training text
  almost never shows one alone, while 5.5% of Mozhi's training words are one. M3 checks how its line finder
  cuts lines against how the training crops were cut.
* **CPU speed is the open cost.** On the GPU the CRNN outreads Tesseract on lines and words; on
  the CPU alone it is slower on words and receipt lines (288 against 511 Mozhi words per second).
  M6 serves on CPUs, so ONNX export and batching by page are measured there.
* **On the Apple GPU, memory follows tensor shapes.** Five fixed batch shapes (decision 21) held
  GPU memory at 5.3 to 5.6 GB through both 20,000 step runs. Later models trained on the Mac keep
  their shapes fixed the same way.

## Decisions in M3

**28. Test pages: synthetic pages and CORD's receipts.**
What: 300 synthetic pages (30 pages of text, Hindi, English and mixed, in one and two columns, each
set once with the training fonts and once with the six held out fonts, clean and under four page
damages) and CORD's 100 test receipts.
Why: measuring line finding needs every line's position, text and order. No open set of printed
Hindi pages marks its lines: Mozhi publishes word crops only, and IndicDLP marks layout blocks
(paragraphs, tables, headers) without lines or text. Synthetic pages give exact lines and order and
change one thing at a time (script, layout, fonts, damage); CORD gives real photographs.
Why not real scanned Hindi books: Wikisource's proofread scans have text but no line boxes, so they
could only give page CER, and they would need downloading on the Mac; left out to keep M3 short.
Why not FUNSD or SROIE: English only, with licences that limit use.

**29. CORD's receipts masked to their annotated part.**
What: everything outside the hull of a receipt's annotated words (widened by a word height), and
CORD's separator lines and do not care regions, is painted in the paper's colour before any reader
sees the receipt. Its lines are CORD's rows, each the smallest rectangle around its words, top to
bottom. Runs of three or more of one separator character (------, =====) are removed from both
texts before scoring.
Why: CORD annotates the item list but not the shop's details (blurred for privacy) nor every
separator; unmasked, every reader would be charged for reading real text that has no truth. The
mask is the same for every reader.

**30. Pages drawn like M1's lines.**
What: A4 at 300 dpi, a title, one or two columns of paragraphs on a baseline grid, a page number;
every line drawn by render_line, its truth the tightest box around its ink, carried through the
damage as a quadrilateral. Damage: clean, scanner noise, a shadow, the page turned by 1 to 4
degrees, a phone photo (perspective, uneven light, soft focus, JPEG). Text: Hindi Wikipedia
documents of TinyGPT's validation split and held out English sentences; validation pages never
share a document or a sentence with test pages, and the seen and unseen font sets share their text.
Why: 300 dpi is how books are scanned and what Tesseract expects; one renderer keeps Devanagari
shaping identical across milestones; the same text in both font sets compares like for like. The
page text comes from the documents M2's validation lines came from, so a few sentences may overlap;
those lines only chose M2's checkpoint.

**31. Three line finders, and the true lines.**
What: Tesseract's own page analysis (page mode 3); a classical finder written for DocLens (decision
35); docTR's pretrained DBNet (ResNet 50) with docTR's own grouping of words into lines and its
reading order, also scored in docTR's default order (top to bottom, left to right); and the true
lines. Every finder's lines are read by the same recognizer (decision 33); Tesseract is also scored
reading its own lines.
Why: the true lines show what reading alone costs, so the rest of each finder's error is finding;
Tesseract is the baseline of the whole project; the classical finder is computer vision from first
principles; docTR is a learned detector. Why docTR's DBNet: Apache 2.0, PyTorch, a standard detector,
ready to use. Why not train a detector of our own: hours of training for a part where a strong open
one exists; DocLens's from scratch model is its recognizer. Why not Surya or PaddleOCR here: they are
compared whole in M5.

**32. Lines are quadrilaterals; Tesseract's are rebuilt from its words.**
What: every line, true or found, is four corners at any angle. Tesseract's line is the smallest
rectangle around the words it put in the line.
Why: on a turned page an upright box covers parts of the lines above and below. On a page turned by
a few degrees, Tesseract's own upright line boxes matched 7 of 32 true lines while it read the page
with 1.5% CER; rebuilt from its words, 31 of 32 matched.

**33. One recognizer for every finder: M2's CRNN.**
What: greedy decoding; each line cut out along its quadrilateral, turned level, with 15% of its
height of page around it; a line wider than 24 times its height is cut at the gap between words
nearest its middle and the pieces' readings joined.
Why greedy: in M2, beam search with TinyGPT changed the main model's CER by at most 0.13 points, and
the page sets hold about 15,000 lines. Why level crops: skew was the CRNN's weakest damage in M2.
Why cut long lines: M2 trained on lines up to 24 times as wide as high, and half the lines of a one
column page are wider; squeezing them would distort every letter.

**34. Page scores.**
What: page CER on text with every run of spaces and line breaks made one space; line precision,
recall and F1 at an IoU of 0.5 between quadrilaterals, one to one; splits and merges; Kendall's tau
of the reading order of matched lines; seconds per page; bootstrap intervals over pages and a
paired comparison with Tesseract.
Why page CER: it is what a reader of the page gets, and it charges a finder for missed lines,
merged lines (read badly) and a wrong order, while a line cut in two costs nothing if its pieces
are read in order. Why IoU 0.5: ICDAR's text detection protocol. Why splits, merges and tau: they
say why a finder fails.

**35. The classical line finder.**
What: Sauvola binarisation; the page levelled by M1's skew estimate; components, without specks,
frames and separator dashes; a recursive XY cut over the components that cuts columns first,
searches gaps between columns at slight slants, cuts columns only between sides of about the same
width with a gap much narrower than they are, and cuts off a title or a page number that reaches
into the gap; in each block, lines from the rows of ink, each block levelled again by its own angle
(doclens/layout.py).
Why: the XY cut is the classic, explainable way to find blocks, and its cut order is a reading
order. The column rules come from what failed on the validation pages and receipts: a receipt's
prices beside its item names looked like a column; photographed pages tilt their columns by
different angles; a centred title blocks the gap between the columns below it; a paragraph's short
last line has too little ink to count as a line against a page wide threshold. Settings (Sauvola's
k 0.1 and window of 3 text heights, a closing of 3 pixels) were chosen on the validation pages and
CORD's validation receipts by the mean of their line F1, never on a test set: 0.999 on the pages and
0.59 on the receipts.

**36. Two faults found in the first run, fixed, and every page drawn again.**
What: the first full run, on pages drawn with seed 0, found two faults. (1) The classical finder's
rule for a title or a page number in the gap between columns could cut off a strip many lines tall
as if it were a title (36 line pitches on one page), and the columns were then read in strips: the
top of both, then the bottom of both. Now only a strip at most 4 line pitches tall can be cut off
this way; two title lines at the largest title size are 3.4. (2) A word wider than its column, a
web address on one page design, was drawn whole, across the column beside it. Now such a word is
broken between letters, without a hyphen, as web addresses are broken in print. The validation
pages grew from 6 page designs to 30, and every set was drawn again with seed 2 (the validation
pages with seed 3): new layouts, new damage and a new choice of text and fonts for every page, from
the same held out text and fonts. The first
run's results are kept in `docs/results/m3_pages_first_run.md`. The evaluation now reuses a cached
result only for the same page image and the same finder code, so redrawn pages are never scored
with old lines.
Why: fixing faults found on test pages and scoring those pages again would measure the fixes on
the pages they were made for; fresh pages measure them fairly. The limit of 4 pitches comes from
the page design (a title has at most two lines), not from the test pages, and was checked on the
new validation pages and CORD's validation receipts before any test page was scored. On the 150
validation pages the old rule read 3 two column pages in the wrong order and the new rule none; the
lines changed on those 3 pages only, and page CER went from 1.09% to 0.74%. On the 100 validation
receipts the lines changed on 1 receipt and page CER stayed at 33.3%.
Why not keep the old pages and leave out the flawed design: words wider than a column can turn up
on any page drawn from the same text, and the finder's rule was wrong on four other designs.

**37. Intervals over page designs, not over pages.**
What: on the synthetic sets, bootstrap intervals and paired comparisons resample page designs, each
with its five damaged pages, instead of single pages; on CORD every receipt is a design of its own,
so nothing changes there.
Why: a design's five pages share their text, layout and fonts, and a finder that fails on a layout
fails on most of its five pages (the first run's broken pages came in whole designs), so 150 pages
are not 150 independent draws. Over single pages the intervals were too narrow: docTR, unchanged
between the runs, read the first run's pages with seen fonts at 6.6% (4.3% to 9.3% over pages) and
the new ones at 10.9%. Over designs its interval on the new pages is 5.1% to 17.4% (7.7% to 14.5%
over pages), while Tesseract's barely moves (8.6% to 14.2%, against 8.6% to 14.1%): the intervals
widen exactly where a finder depends on the layout. Why not over pages: that answers how a finder
would do on these same 30 designs under new damage, not on new pages.

## What M3 found, and what it means for M4

The full tables are in `docs/results/m3_pages.md` (the first run's in `m3_pages_first_run.md`),
the headline numbers and findings in the README.

* **The error left on a page is in finding its lines.** The CRNN reads the true lines at 0.1% and
  0.4% CER on the synthetic pages and 1.9% on CORD's receipts; the best finder on each set leaves
  0.4%, 1.6% and 5.6%. Whole pages now get better through better finding, not a better recognizer.
* **For printed pages the classical finder is the one to keep.** 0.4% and 1.6%, against
  Tesseract's 11.4% and 9.7% and docTR's 10.9% and 12.0%, at 0.21 s a page on the CPU. In the
  unseen fonts, over half of what is left is on photos of two column pages (8.2%, with merged
  lines), the case to watch when M5 compares whole systems.
* **M4 reads receipts with docTR's detector in its top to bottom order.** On CORD that gives 5.6%
  page CER, against 51.8% for Tesseract and 44.2% for the classical finder, whose rules were
  written for pages. M4 extracts fields from the true text and from this OCR text, so what
  reading errors cost the fields is measured, not assumed.
* **Line F1 can mislead, so M4 is scored end to end.** docTR has the lowest line F1 on the
  receipts (13.8: it finds names, quantities and prices as separate pieces where CORD has one row)
  and the best page CER. M4 reports field F1 on OCR text, not detection scores.
* **Reading order must fit the document.** docTR's reading order is needed for columns (10.9%
  against 44.5% top to bottom) yet still breaks on 26 and 28 of the 75 two column pages, and it
  is wrong for receipts (47.1% against 5.6%), where it takes the prices for a column. The
  classical finder's order is right on both (tau 1.000 and 0.998 on the pages, 1.000 on the
  receipts), but on receipts its lines are not. A service for both kinds of document must tell
  them apart or use an order that handles both; decided when the service is built (M6).
* **The fixes held on fresh pages (decision 36).** On these test pages the old title rule would
  have read 12 and 15 of the 75 two column pages in the wrong order; the new rule reads 0 and 1.
  Settings are now chosen by page CER, which sees reading order, not by line F1 alone, which
  missed the fault on the first validation pages.
* **docTR's detector needs level pages.** On pages turned by 1 to 4 degrees it cuts 49% to 63% of
  the lines into pieces, and on turned two column pages its reading order then fails (38.8% and
  47.4%). Levelling each page with the skew estimate before docTR sees it is the fix to try, in
  M5.
* **Level crops settled skew.** Cut along their quadrilaterals and turned level, the true lines
  of turned pages read as well as those of clean pages (0.1% and 0.4%), so training on more skew,
  planned after M2, was not run.
* **Which designs are drawn matters.** docTR, unchanged, read the first run's pages at 6.6% and
  these at 10.9% (seen fonts). Intervals over designs (decision 37) show that spread; every later
  synthetic set keeps a design's pages together in its intervals.
* **Reading is the slow part.** Per page, finding takes 0.21 s (classical finder, CPU) or 0.33 to
  0.34 s (docTR, GPU) and reading 0.8 to 1.1 s on the GPU, against Tesseract's 11 to 12 s on one
  core. M6 measures reading on CPUs, and batching lines across pages.

## Decisions in M4

**38. Fields: CORD's groups, tagged word by word, in reading order.**
What: CORD's 800 training, 100 validation and 100 test receipts. Every annotated word carries the
category of its group (menu.nm, menu.price, total.total_price and so on), as a BIO tag: B on the
group's first word, I on the rest. Words are read in reading order, CORD's rows top to bottom and
each row left to right, the order docTR's rows give OCR words (M3); the label list holds every
category of the three splits. The receipt image is masked to the annotated part, as in M3.
Why: one tag per word is how LayoutLMv3 and LiLT were evaluated on CORD, so the code can be
checked against their numbers (decision 40). Why reading order and not the order of CORD's JSON:
OCR words have no JSON, and true and OCR words must reach a model in the same order, or the
comparison between them measures the order and not the reading. Why CORD: its items, counts,
prices and totals are what a user wants from a receipt, its split is the one the papers use, and
M1 to M3 already read it.

**39. Three models on one vocabulary.**
What: RoBERTa base (the words alone), LiLT on RoBERTa base (the words and their boxes; MIT, the
model to ship) and LayoutLMv3 base (words, boxes and the image; its weights are CC BY NC SA 4.0,
so it is for comparison only). All three use RoBERTa's byte level BPE vocabulary, so one
tokenizer turns a receipt into the same tokens for each; scripts/bench_m4.py checks that LiLT's and
LayoutLMv3's own vocabularies are RoBERTa's. A word's tag is read from its first token.
Why: LiLT's text stream starts from RoBERTa base's weights, so LiLT against RoBERTa isolates what
the boxes add, and LayoutLMv3 against LiLT what the image adds. Why base and not large: three
models and several seeds must train on one Mac. Why not LayoutLMv2 or LayoutXLM: older and need
detectron2; why not an OCR free model (Donut): whole systems are compared in M5.

**40. Training: fixed lengths, 0.00005 for 2,000 steps, the best validation field F1 kept, three
seeds; the papers' boxes only as a check.**
What: receipts are cut into windows of whole words of at most 384 tokens and padded to 128, 256
or 384 tokens, so the Apple GPU sees three shapes (decision 21); batches of 8 windows of one length,
the windows left over at a length carried up to the next, so every window trains once an epoch.
LayoutLMv3 takes each batch in two parts of 4 whose gradients add up before the step, weighted by
their labelled tokens, so the step is the one the whole batch would take; it validates in parts of
4 too, and the GPU's cache is emptied after every validation.
AdamW at a learning rate of 0.00005, a linear warmup over the first 10% of 2,000 steps (20 epochs)
and a linear decay to zero; every 100 steps the model tags the validation receipts and the weights
with the best field F1 are kept; three seeds per model (scripts/run_m4.sh). Every word gets its
own box. Once per layout model, the papers' setting is run as a check: every word given its whole
field's box (segment boxes), on which LayoutLMv3 base reports 96.56 entity F1 and LiLT 96.07.
Why: scripts/bench_m4.py found CORD's training receipts short (762 of the 800 fit 128 tokens, the
longest 296), so 384 tokens hold every one, and the largest GPU pool it saw, 11.7 GB for LayoutLMv3
at 512 tokens, is a reason to stop below 512. Even so, LayoutLMv3's first run ran out of GPU memory
after 100 steps with whole batches of 8: its image adds 197 positions to every window, and attention
grows with the square of the length. Parts of 4 keep the batch of 8 the other models train with,
at the cost of a little speed. Hugging Face's example for LayoutLMv3 on CORD trains
at 0.00005, scores every 100 steps and keeps the best model by F1, for 1,000 steps; 2,000 give LiLT
and RoBERTa room, and the best validation step is kept anyway. The benchmark measured 0.38, 0.46
and 1.09 s a step at 128 tokens for RoBERTa, LiLT and LayoutLMv3 (13, 16 and 37 minutes for 2,000
steps), so three seeds of each fit in a night. Why three seeds: runs differ (LayoutMask's ablation
reports spreads of 0.12 to 0.45 points between runs on CORD and FUNSD), and a difference between
models should be larger than that. Why the field's box only as a check: it tells the model where
each field starts and ends ("every entity in FUNSD and CORD exactly shares the same segment", from
LayoutMask's ablation, where word boxes cost that model 0.7 points of F1 on CORD and 9.4 on FUNSD),
and no OCR knows a receipt's fields before they are found.

**41. Test inputs: the true words and the OCR words; a field counts only when its text is exact.**
What: every model reads the 100 test receipts twice: CORD's true words and boxes, and the OCR
words, docTR's detector words read by M2's CRNN in docTR's top to bottom order (M3's best reader of
receipts, 5.6% page CER; scripts/prepare_m4.py). Field F1 counts, per receipt, the (category,
text) pairs found in both the truth and the prediction, texts compared without spaces; a bootstrap
over receipts gives the intervals and pairs models on the same receipts. Two ceilings: perfect tags
on the true words (99.7%, not 100: CORD sometimes makes a separator such as ":" a field of its own
inside another field of the same category), and on the OCR words with the tags of the true words
under them, which is what the OCR alone costs.
Why exact text: a total with one wrong digit is a wrong total. Why without spaces: OCR and CORD
cut a line into words differently ("SubTotal" against "Sub Total"). Why not seqeval's entity F1:
it needs the true words, so it cannot score OCR text; field F1 scores both inputs the same way.
Why also train on OCR words: models train on true words, as in the papers, and LiLT is also trained
on the OCR words of the training receipts (their tags moved from the true words under them, three
seeds), to see whether training on what the OCR gives closes the gap between true and OCR words.

**42. What layout adds, and why a model decides.**
What: LiLT and LayoutLMv3 are also scored with every word given the same box (the whole receipt),
and LayoutLMv3 with a blank image; field F1 per category; each run's validation and test F1.
Why: removing an input at test time shows how much a trained model leans on it, without training
another model; per category F1 shows which fields layout helps (a price beside its item, a total
at the bottom).

## What M4 found, and what it means for M5

The full tables are in `docs/results/m4_fields.md` and `docs/results/m4_ocr_losses.md`, the
headline numbers and findings in the README.

* **Ship LiLT, on words in reading order.** On true words the three models are within half a point
  (RoBERTa 93.4%, LiLT 93.9%, LayoutLMv3 93.7% field F1; no difference measurable), and on OCR
  words LiLT is 0.9 points above RoBERTa (p 0.01) and not measurably below LayoutLMv3 (p 0.22).
  LiLT is MIT, trains in 18 to 21 minutes where LayoutLMv3 takes 44, and does not need the image,
  which LayoutLMv3 does not use anyway (a blank image changes nothing, p 0.26).
* **The OCR is where fields are lost, so M5 compares OCR systems by the fields they give.** From
  true to OCR words every model loses 13.5 to 14.5 points, and perfect tags on the OCR words reach
  only 85.2%: 208 of the 1,309 test fields are lost before any tagging. M5 runs each OCR system it
  compares through the same LiLT and reports field F1 next to page CER, because a page CER says
  little about whether the total is right.
* **Most lost fields are one character off.** 116 of the 208 differ from the truth by a single
  character, 70 of them in a digit or a separator. Receipts check themselves (item prices add up
  to the subtotal, the subtotal and tax to the total, cash minus total to the change), so a check
  of the arithmetic could find, and sometimes fix, a misread number; measured when the service is
  built (M6).
* **The order of the words carries the layout.** A text model reading CORD's words in reading order
  does as well as the layout models, which still lean on their boxes when they have them (one box
  for every word costs LayoutLMv3 35.5 points, LiLT 9.0). The order rule found in M3 is what the
  service must keep; on documents whose order is harder to find than a receipt's, boxes may matter
  more, which CORD cannot show.
* **Published numbers lean on the truth's boxes.** With every word given its field's box, LiLT
  gets 96.4% (its paper 96.07) against 93.9% with word boxes; DocLens reports word boxes, the ones
  an OCR can give.
* **Training on OCR words is not worth its cost here** (+0.7 points on OCR words, p 0.13, and 0.7
  fewer on true words), and validation scores flatter by 1.2 to 2.5 points, so test sets are scored
  once, after every choice is made.

## Decisions in M5

**43. Six systems, each run the way its makers show it, each in its own environment.**
What: DocLens (M2's CRNN, M3's line finders, M4's extractors); Tesseract 5.5, as in M1 and M3;
PaddleOCR 3.7 with the models it picks for the language it is told (for English PP OCRv6's medium
detector and recognizer, for Hindi PP OCRv5's server detector and its Devanagari recognizer) and
without its document orientation, unwarping and text line orientation models, as in its quick
start; EasyOCR 1.7 (its CRAFT detector and its recognizer for the languages given); PaddleOCR VL
1.6, the whole pipeline: its layout model (PP DocLayoutV3) on the CPU finds and orders a page's
blocks, and its vision language model of 0.96 billion parameters, served on the Apple GPU by MLX VLM
as its Apple silicon guide sets it up, reads each block; Surya 0.22, whose model Surya 2 (a vision
language model of about 0.6 billion parameters) runs in llama.cpp's server on the Apple GPU, eight
requests at a time, with Surya's default full page OCR. Every system has its own virtual
environment with pinned versions (scripts/setup_m5.sh) and a small runner (scripts/m5/) that
writes what the system read, the time it took and its peak memory, in one shape for all.
Why: a comparison is fair only when each system runs as its users would run it, on the same
machine. Separate environments because their pins clash: Surya wants Pillow below 11 and one exact
OpenCV, MLX VLM a newer OpenCV, PaddleOCR its own framework. One output shape, so the scoring cannot
differ between systems. PaddleOCR VL's whole pipeline because its guide warns that the model alone,
run through Transformers, is not the system its published numbers come from. PaddleOCR's three
preprocessing models are for photographed, warped or upside down documents and its quick start
turns them off; every M5 page is upright.
Why not larger vision language models (olmOCR and Qwen2.5 VL at 7 billion parameters, dots.ocr at
about 3): too large or too slow for a 16 GB Mac; PaddleOCR VL and Surya 2 are the small document
models that run on one. Why not PaddleOCR's PP StructureV3 for pages: PaddleOCR VL is Paddle's
system for whole documents here, and the plain pipeline is what most people call PaddleOCR. Why
not docTR as a whole system: its recognizers read Latin script only; its detector is already part
of DocLens.

**44. The same images and the same truth for every system, and the language a careful user would
give.**
What: scripts/m5_export.py writes every image once, exactly as M1 to M4 read it (grey PNG): a fixed
random sample (seed 0) of 1,000 Mozhi test words, 1,000 synthetic test lines in the six held out
fonts, 1,000 CORD test words and all 791 CORD test lines; the 150 synthetic test pages in the held
out fonts (30 designs, each clean and under four kinds of damage); and CORD's 100 test receipts,
masked as in M3. Tesseract, PaddleOCR and EasyOCR are told the language as in M1: Hindi for Mozhi,
English for CORD, both for the synthetic lines and pages. DocLens and the two vision language
models are told nothing.
Why a sample of the crops: the vision language models read crops one request at a time, so the
24,784 crops of the M1 sets would take each of them a night; 1,000 per set keep the intervals to a
fraction of a point at these error rates, and since every system reads the same images, systems
are compared in pairs. Why the held out fonts: DocLens trained on the other fonts; the other
systems saw none of ours. Why grey: these are the pixels every earlier milestone scored.
Why not the training fonts too: they would flatter DocLens alone.

**45. Scored as before: M1's CER, M3's page CER, M4's field F1, on the images every system read.**
What: CER per set (NFC, every run of spaces made one) with bootstrap intervals over images and a
paired comparison with DocLens; page CER in each system's own order, with intervals over page
designs on the synthetic pages (decision 37) and the page CER by layout and damage. On the
receipts, field F1 through M4's extractors trained on true words, three seeds pooled: RoBERTa on
each system's words in the order the system writes them, and LiLT on its words and their boxes put
in the order DocLens reads receipts (docTR's rows), wherever the system gives boxes for lines or
words, a line's box shared among its words by their length; beside them, the best any tagger could
do on those words (the true fields moved onto them, as in M4). An image a system failed on counts as
an empty reading. Surya writes every block as HTML, and PaddleOCR VL writes a table as HTML (on
receipts its layout model often finds one); both become plain text before scoring, a line for each
block, row or line break and a space between table cells.
Why the system's own order for page CER and RoBERTa: it is what a user of the system gets. Why rows
for LiLT: M4's extractors learned words in that order, and putting every system's words in the same
order measures their reading, not their ordering. Why share a line's box: PaddleOCR, EasyOCR and
PaddleOCR VL box lines, not words, and LiLT needs a box per word. M4 kept a docTR word read with a
space inside as one word; here every system's text is split at spaces, so DocLens's field F1 can
differ a little from M4's.
Why not line F1: the vision language models give blocks, not lines, and M3 showed that line F1 can
mislead (decision 34).

**46. Speed and memory as each system runs on the Mac.**
What: seconds per image from each runner's own clock, loading left out, with each system's own
batching: DocLens 64 crops at a time on the GPU, Tesseract one process per core, PaddleOCR 16 crops
at a time on the CPU (PaddlePaddle has no Apple GPU backend), EasyOCR one at a time on the GPU,
Surya eight requests at a time, PaddleOCR VL through MLX VLM. Memory is the peak physical footprint
of the runner's process plus that of the server holding the model; Tesseract's is its largest
single process.
Why: speed and memory decide what can be shipped on a laptop or a small server, and each system as
it runs is what its users get. Why the physical footprint (macOS's own figure, the one Activity
Monitor shows): on Apple silicon the CPU and the GPU share memory, and the resident size misses
what the GPU holds for a process.
Why not one batch size for all: each system's default is what its users get; one size would slow
some systems and break others.

**47. Licence beside accuracy.**
What: the tables give each system's licence for its code and its weights. Tesseract, PaddleOCR,
EasyOCR and PaddleOCR VL are Apache 2.0, code and models. Surya's code is Apache 2.0, but Surya 2's
weights come under the AI PUBS OpenRAIL M licence, modified: no use by an organisation with more
than $5 million of revenue or funding, nor in a product that competes with its maker's, except for
personal use and research. DocLens is MIT, with docTR's detector (Apache 2.0) for receipts.
Why: a system that cannot be shipped is not an option for a product, however accurate.

**48. Two settings the Mac forces, measured before the full run.**
What: (1) PaddleOCR's detector sees a page at most 1,600 pixels a side; the recognizer still reads
each line cut from the full page. (2) PaddleOCR VL keeps the pipeline its Apple silicon guide sets
up, its model served by MLX VLM, and a check beside the comparison reads 200 Mozhi words, 200
synthetic lines and 100 CORD lines again with the model in Hugging Face Transformers, its reference
implementation, in float32 on the CPU (scripts/m5/run_paddleocr_vl_ref.py), each crop sized as the
pipeline sizes a text block. scripts/m5/probe.py measured both choices on the Mac first.
Why (1): by default PaddleOCR detects on a page at its own size (up to 4,000 pixels a side), and
on the Mac the first synthetic page was killed for lack of memory. PaddlePaddle for macOS is built
without oneDNN (its build reports MKL off and ships no oneDNN library), so each convolution first
unrolls the whole image, and one 9 by 9 convolution over 256 channels in the neck of PP OCRv5's
server detector, at a quarter of a 300 dpi page's size, needs about 45 GB for that alone. On a
synthetic page of 35 lines, limits of 960, 1,280 and 1,600 pixels all found the 35 lines, with page
CER 1.8%, 1.7% and 1.8%, peaks of 6.2, 8.3 and 10.0 GB and 17.7, 21.9 and 25.8 s. The largest
is used: it fits in the Mac's 16 GB and leaves 79 of CORD's 100 test receipts at their own size.
The second smoke run then showed one process growing to 16.2 GB over three synthetic pages and
three receipts: on the Mac PaddlePaddle takes CPU memory from a buddy allocator that keeps what it
took, and pages of other sizes asked for more. So each set of pages is read in a process of its
own, largest pages first, so later pages fit in what the first took. Even so, the full run's page
processes peaked at 18.3 GB, more than the Mac's 16 GB: macOS compressed and swapped memory to hold
them, and a synthetic page took 28.9 s against the probe's 25.8 s. The size was kept, because
results already measured at it would otherwise have to be thrown away for a smaller one that the
probe found no more accurate; the memory is reported as measured.
Why not 960 (PaddleOCR 2's default for this setting): it saved memory and time but no accuracy on
the probe's page, and would shrink 73 of the 100 receipts, whose print is small. Why not the mobile
detector: the server detector is PaddleOCR's own choice for Hindi; only its input size had to change.
Why (2): the first smoke run read clean Hindi crops as nonsense, and the probe found two causes.
The model itself misreads Mozhi's words in every setting, the reference included: 57% to 66% CER
on 10 words, "था" read as "41" by all six. How the Mac serves it costs the rest: on 10 synthetic
lines the pipeline through MLX VLM reached 40% CER with the JPEG it sends and 27% with PNG,
enlarging the crop first changed nothing, while the model in Transformers read the same lines at
10% in float32 on the CPU and 16% in bfloat16 on the Apple GPU; on 10 CORD lines every setting was
within a point (2.8% to 3.4%). So the comparison runs PaddleOCR VL as its makers set it up on a
Mac, which is what a Mac user gets, and the check measures on 500 crops how much of its error the
serving path adds, so the model is not blamed for it.
Why not the reference implementation for everything: it is the model alone, without the pipeline's
layout step, which PaddleOCR's guide says is not the same system; it took 3.3 s a crop on the CPU
against 0.6 s, which over the pages would take days; and on the Apple GPU, in bfloat16, it no
longer reads as the reference does. Why not send PNG: that changes the pipeline as shipped, and on
10 lines it closed only part of the gap.

## What M5 found, and what it means for M6

The full tables are in `docs/results/m5_compare.md`, the diagnoses in `docs/results/m5_diagnostics.md`,
the headline numbers and findings in the README.

* **Ship DocLens's own pipeline.** Its CRNN reads crops best or level with the best on three of the
  four sets (Mozhi words 0.5%, CORD lines 2.8%, CORD words 1.2% against PaddleOCR's 1.4%, p 0.38) and
  is second only to Surya 2 on synthetic lines (1.0% against 0.7%); on receipts no system gives
  measurably more correct fields (DocLens 78.9% through RoBERTa, 79.7% through LiLT; PaddleOCR's
  words through LiLT 82.4%, p 0.06); and it reads a page in 1.1 s and a receipt in 0.37 s, where
  the modern systems take 21 to 41 s a page and 2.5 to 13.7 s a receipt. It is MIT licensed and
  its models are small.
* **Surya 2 is the stronger reader of whole pages, but not one to ship here.** 0.4% page CER
  against 1.6%, and 0.4% on photographed pages where DocLens reaches 4.7%; but 31 times slower,
  14.9 GB with its server, and weights a company above $5 million of revenue may not use. Photos of
  two column pages remain DocLens's weak spot (decision 36 and M3's findings).
* **Reading order decides whole pages.** PaddleOCR and EasyOCR read two column pages straight
  across: 75.1% and 76.2% page CER, against 2.0% and 5.7% with their own lines read column after
  column. The service keeps DocLens's line finder for pages, whose cuts are its reading order.
* **A reader that cannot invent text is worth something.** Both vision language models wrote text
  that was not in the image: Surya 2 on 43 of 1,000 CORD words and 84 of 791 CORD lines,
  PaddleOCR VL on 14 of the 150 synthetic pages, all of them noisy, turned or photographed; a CRNN
  can only write what lies along its line. On a receipt, an invented line is worse than a misread
  one, because nothing marks it.
* **Checking the sums is still the cheapest guard on fields.** Every system's best field F1 stays
  below 83%, and M4 found most lost fields one character off; M6 measures what a receipt's own
  arithmetic catches.
* **Measure serving paths, not just models.** PaddleOCR VL served by MLX VLM on the Mac read
  synthetic lines at 14.7% against 11.2% in its reference implementation (p 0.03), so M6 checks
  DocLens's ONNX graph against the PyTorch model before it ships and measures the service end to
  end over HTTP.

## Decisions in M6

**49. One service, two kinds of document, and the caller says which.**
What: a FastAPI service (doclens/service.py) over one pipeline (doclens/pipeline.py). POST /page
takes a printed page and returns its lines in reading order, each with its text and its four
corners: M3's classical finder, then the CRNN. POST /receipt takes a receipt and returns its words
in reading order (docTR's detector, its rows top to bottom), the CRNN's reading of each, the fields
LiLT tags (M4) and whether each of the receipt's sums holds. GET /health names the models, GET /
serves the demo page. Images come as a file in a form, in any format OpenCV reads, and are read in
grey; empty files, files that are not images, images under 16 pixels a side, over 25 MB or over 60
million pixels are refused with a message. One image is read at a time.
Why the caller says the kind: pages and receipts need different finders and orders (M3: docTR's
order reads a receipt at 47.1% page CER, top to bottom at 5.6%; the classical finder reads pages at
0.4% to 1.6%, receipts at 44.2%), and a wrong guess costs a whole document while asking costs the
caller one word. Why not guess the kind: a classifier would need its own labelled data and test,
and its mistakes would reach the caller as unexplained bad text. Why grey: every number in M1 to M5
was measured on grey images. Why one image at a time: ONNX Runtime and PyTorch already spread a step
over the cores, so parallel requests would only compete; the benchmark found 0.76 receipts a second
from one client and 0.73 from four.
Why FastAPI: typed parameters, multipart uploads and an OpenAPI page for free; why not Flask: none
of those without extensions.

**50. The CRNN in ONNX Runtime on the CPU; docTR's detector and LiLT in PyTorch on the CPU.**
What: scripts/export_models.py exports the CRNN with PyTorch's current exporter (torch.export, then
onnxscript), the batch and the width left free, and checks the graph against the PyTorch model on
every eighth M5 crop before keeping it: all 474 crops read the same, log probabilities within 0.0005. LiLT is the seed of M4's three with the best
validation field F1 (seed 1, 96.2% on the validation receipts), saved with its labels and RoBERTa's tokenizer, so the service
downloads nothing.
Why ONNX for the CRNN: it reads every line, so it is the step that runs most; ONNX Runtime ran it
in 3.3 s against PyTorch's 3.6 s for those 474 crops on the Mac's CPU (1.8 times faster on an x86
Linux CPU), with the same text, and the graph runs wherever ONNX Runtime does. Why the CPU: a service runs on servers without
Apple's GPU, and M2 to M5 measured the GPU already. Why not int8 weights: ONNX Runtime's int8 copy was smaller (11.6 MB against 15.8), faster on the Mac (2.3 s against 3.3 s) and as accurate (1.56% CER against 1.57%), but on an x86 Linux CPU it ran 5.7 times slower than float32; how fast it is depends on the server's CPU, so the float32 graph ships and int8 stays a measured option. Why not docTR
and LiLT in ONNX too: they run once a receipt where the CRNN runs once a line; exporting them would
let the image drop PyTorch, left for later. Why the older exporter was dropped: PyTorch marks its
options deprecated; the current one needs onnxscript, and in PyTorch 2.14 a second export in one
process fixed the width, so export_models.py exports once per process and the export refuses a
graph whose batch or width is fixed.

**51. A receipt's sums are checked and reported, never used to change a field.**
What: doclens/checks.py reads each amount as written (thousands separators and cents, "60.000" and
"60,000.00" alike) and checks three rules on the fields found: the items, their add ons and
discounts add up to the subtotal (or the total when there is none); the subtotal with service,
discounts and tax gives the total (tax added, or already inside); cash less the total is the
change. The service reports each rule that applies and whether it holds. scripts/eval_checks.py
measures them on CORD's 100 test receipts: on the true fields the rules hold on 61 of 64, 57 of 63, 30 of 31 and 47 of 50 receipts (rounding and printing differ); on LiLT's fields from the OCR words (three seeds), a failing rule flags 73 receipts, 62 of them (85%) with an amount misread or lost, but catches only 62 of the 182 such receipts (34%).
Why: M4 found most fields the OCR loses one character off, often a digit, and receipts state their
own sums. Why report and not fix: a one character fix that makes a failing sum hold is found
19 times on LiLT's fields and right 6 times, and field F1 goes from 79.8% to 79.7% (p 0.81); with perfect tags, 6 fixes and 2 right. A failing sum says that an amount is wrong, not which. Why not more rules (quantity times unit price): CORD marks unit prices on too few items
to test them.

**52. One Docker image for the CPU with everything inside, built and checked by CI.**
What: Python 3.13 slim, PyTorch's CPU build, the package with its service extras, docTR's detector
weights fetched while the image is built, models/ copied in, a user without root, a health check;
no network needed when it runs. CI builds the image with tiny untrained models
(scripts/tiny_models.py) and runs scripts/check_service.py against it over HTTP.
Why everything inside: the image reads documents offline with exactly the models measured, and
cannot be started with the wrong ones. Why tiny models in CI: the real ones come from training runs
on the Mac that are not in the repository; CI checks the image's plumbing, the Mac its accuracy.
Why not mount the models at run time: possible (docker run with a volume), but then the image and
its models can drift apart. Why the CPU build of PyTorch: the CUDA build is several times larger
and the service uses no GPU.

**53. The service measured end to end over HTTP, on the Mac's CPU.**
What: scripts/bench_m6.py starts the service, runs the end to end checks, then sends it the 150
synthetic pages and 100 receipts M5 compared every system on, one at a time, then the receipts from
1, 2 and 4 clients at once; it scores what came back as M5 scored DocLens and records the service's
peak memory: a page waits 1.42 s at the median (finding 0.15 s, reading 1.20 s) and 1.95 s at the 95th percentile, a receipt 1.23 s and 1.46 s; the service reads the pages at 1.6% page CER and the receipts at 6.0% with 79.6% field F1, as M5 measured DocLens on the GPU (1.6%, 6.0%, 79.7% over three seeds); 0.76 receipts a second from one client and 0.73 from four; 3.3 GB at its peak, with a models folder of 516 MB.
Why: what a user waits for includes HTTP and decoding the image, and scoring the service's answers
checks that the shipped path (ONNX, the CPU) reads as the measured one did.

**54. A demo page in plain HTML, served by the service.**
What: one page (doclens/demo.html) at /: choose or drop an image, say whether it is a page or a
receipt, and see the lines or words drawn over the image, the text in reading order (a line and its
box light up together), and for a receipt its fields and its sums.
Why plain HTML: it ships inside the package and the image, needs no build step and no network.
Why not run the models in the browser (ONNX Runtime Web): the CRNN could, but the line finder and
docTR are Python, so the page would read single lines only.

## What M6 found

The tables are in `docs/results/m6_service.md` and `docs/results/m6_checks.md`.

* **The shipped path reads as the measured one did.** Through HTTP, ONNX Runtime and the CPU the
  service reads the synthetic pages at 1.6% page CER and the receipts at 6.0%, with 79.6% field F1,
  the numbers M5 measured for DocLens on the GPU.
* **A page in under 1.5 s and a receipt in about 1.2 s on a laptop CPU,** in 3.3 GB. Reading the
  lines is most of a page's time (1.20 of 1.42 s), finding the words most of a receipt's (0.77 s);
  more clients do not add throughput, one image at a time already uses the cores.
* **Sums flag, they do not fix.** A failing sum points at a receipt with a wrong amount 85% of the
  time, but finds only a third of them, and changing a digit to make a sum hold is right a third of
  the time; the service reports the sums and leaves the fields as read.

## After M6: CI made to pass

**55. Type checks against the newest libraries, as CI installs them.**
What: the type check step had failed on every push since the first M2 commit, so CI's tests and its
image job never ran. The causes were stricter type stubs in the newest releases that CI installs:
numpy 2.5 types `np.float32([...])` as a scalar, OpenCV 5 marks `imread` and `imdecode` as possibly
returning nothing, PyTorch's ONNX exporter returns an optional program, and Transformers' config
classes and doctr's document builder type their arguments. The code now builds arrays with
`np.array(..., np.float32)`, checks every result that can be missing, and types its keyword
dictionaries. Nothing changes at run time (the same arrays and values): the 121 tests pass with the
newest numpy, OpenCV and Pillow, and the type check is clean with the newest PyTorch too.
Why: a check that always fails checks nothing; the type errors hid whether the tests and the image
would pass on a clean machine. Why not pin every version in CI: a pinned CI checks an environment
nobody installs; `pyproject.toml` allows the newest releases, so CI keeps checking those, and a
failure there is an early warning about what a new user would get.

## What runs where

Everything that produces a number runs on the M2 MacBook. M1 needs only the CPU (Tesseract uses
every core); M2 trains on the Apple GPU through PyTorch's MPS backend, with data workers on the
CPU cores. M3 runs Tesseract on every core, docTR's detector and the CRNN on the Apple GPU. M4
trains the extractors on the Apple GPU. M5 runs every system on the Mac in its own environment
(PaddleOCR on the CPU, the others on the Apple GPU). M6 serves and measures on the Mac's CPU. Two
numbers in decision 50 come from an x86 Linux CPU instead, to show that the int8 graph's speed
depends on the processor. GitHub Actions runs the tests on Linux after every push, and builds the
Docker image and checks it end to end.
