# Fonts

The fonts DocLens draws text with. They live in the repository so that every machine renders
the same pixels. `fonts.json` lists each file with the scripts it covers (`deva` Devanagari,
`latn` Latin), its split and its licence:

* **train** fonts draw the training lines (M2) and the `test_seen` set;
* **test** fonts are never used for training and draw the `test_unseen` set, which measures
  how a recognizer copes with letter shapes it has never seen.

Each font keeps its own licence; the full texts, as shipped by Debian and Ubuntu, are in
`LICENSES/`. The MIT licence of DocLens covers the code, not these files.

| File | Family | Scripts | Split | Licence |
|---|---|---|---|---|
| `Aksharyogini2Normal.ttf` | Aksharyogini 2 | deva | train | OFL 1.1 |
| `IBMPlexSansDevanagari-Regular.ttf` | IBM Plex Sans Devanagari | deva | train | OFL 1.1 |
| `IBMPlexSansDevanagari-Bold.ttf` | IBM Plex Sans Devanagari Bold | deva | train | OFL 1.1 |
| `IBMPlexSansDevanagari-Medium.ttf` | IBM Plex Sans Devanagari Medium | deva | train | OFL 1.1 |
| `kalimati.ttf` | Kalimati | deva | train | GPL 2.0 or later |
| `Lohit-Devanagari.ttf` | Lohit Devanagari | deva | train | OFL 1.1 |
| `nakula.ttf` | Nakula | deva | train | GPL 2.0 or later |
| `NotoSansDevanagari-Regular.ttf` | Noto Sans Devanagari | deva | train | OFL 1.1 |
| `NotoSansDevanagari-Bold.ttf` | Noto Sans Devanagari Bold | deva | train | OFL 1.1 |
| `NotoSerifDevanagari-Regular.ttf` | Noto Serif Devanagari | deva | train | OFL 1.1 |
| `NotoSerifDevanagari-Bold.ttf` | Noto Serif Devanagari Bold | deva | train | OFL 1.1 |
| `sahadeva.ttf` | Sahadeva | deva | train | GPL 2.0 or later |
| `Samyak-Devanagari.ttf` | Samyak Devanagari | deva | train | GPL 3.0 or later, font exception |
| `Sarai.ttf` | Sarai | deva | train | GPL 2.0 |
| `Carlito-Regular.ttf` | Carlito | latn | train | OFL 1.1 |
| `DejaVuSans.ttf` | DejaVu Sans | latn | train | Bitstream Vera |
| `DejaVuSerif.ttf` | DejaVu Serif | latn | train | Bitstream Vera |
| `IBMPlexMono-Regular.ttf` | IBM Plex Mono | latn | train | OFL 1.1 |
| `IBMPlexSans-Regular.ttf` | IBM Plex Sans | latn | train | OFL 1.1 |
| `IBMPlexSerif-Regular.ttf` | IBM Plex Serif | latn | train | OFL 1.1 |
| `LiberationMono-Regular.ttf` | Liberation Mono | latn | train | OFL 1.1 |
| `LiberationSans-Regular.ttf` | Liberation Sans | latn | train | OFL 1.1 |
| `LiberationSans-Bold.ttf` | Liberation Sans Bold | latn | train | OFL 1.1 |
| `LiberationSerif-Regular.ttf` | Liberation Serif | latn | train | OFL 1.1 |
| `NotoSans-Regular.ttf` | Noto Sans | latn | train | OFL 1.1 |
| `NotoSans-Bold.ttf` | Noto Sans Bold | latn | train | OFL 1.1 |
| `NotoSerif-Regular.ttf` | Noto Serif | latn | train | OFL 1.1 |
| `OpenSans-Regular.ttf` | Open Sans | latn | train | Apache 2.0 |
| `Roboto-Regular.ttf` | Roboto | latn | train | Apache 2.0 |
| `AnnapurnaSIL-Regular.ttf` | Annapurna SIL | deva | test | OFL 1.1 |
| `AnnapurnaSIL-Bold.ttf` | Annapurna SIL Bold | deva | test | OFL 1.1 |
| `Gargi.ttf` | Gargi | deva | test | GPL 2.0 or later |
| `Caladea-Regular.ttf` | Caladea | latn | test | OFL 1.1 |
| `Lato-Regular.ttf` | Lato | latn | test | OFL 1.1 |
| `Lato-Bold.ttf` | Lato Bold | latn | test | OFL 1.1 |
