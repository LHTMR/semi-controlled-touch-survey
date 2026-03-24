# IASAT_poster_Sarah

This folder contains analysis scripts, figure output, and descriptor map generation for the IASAT poster pipeline.

## Scripts
- `analysis_IASATposter.R`
  - input: `Processed Data/touch_data.txt` parsed from raw survey data.
  - outputs:
    - summary counts: responses per touch and per respondent
    - valence/arousal ranking boxplots by touch descriptors
    - direct comparisons with bootstrapped confidence intervals (valence/arousal)
    - word frequency extraction and plots for questions: `Social_self`, `Social_place`, `Social_context`, `Social_body`, `Intention&Purpose`, `Sensory`, `Emotional_self`, `Emotional_touch`
    - output text files in `Processed Data/`: `*_word-freq.txt`, `*_word-freq-plot-data.txt`
    - plots in `IASAT_poster_Sarah/Figures/`: `*_word-freq.pdf`
    - appropriateness distribution + direct comparisons (`Yes`, `No`, `It depends`, `I don't know`)
- `descriptor_maps.R`
  - input: pattern-matched `Processed Data/*_word-freq-plot-data.txt`
  - maps word stems to descriptors and descriptor groups, including custom grouping logic for attention, person, place, emotion, sensory, intention.
  - outputs `Processed Data/descriptor_map_data.tsv`
  - generates descriptor map plots in `IASAT_poster_Sarah/Figures/`: `sensory_map.svg`, `person_map.svg`, `affect_map.svg`, `place_map.svg`, `intention_map.svg`.

## Data
- `Processed Data/descriptor_map_data.tsv`
- `Processed Data/*_word-freq.txt`
- `Processed Data/*_word-freq-plot-data.txt`
- `IASAT_poster_Sarah/Figures/` contains subfolders `Appropriateness`, `Valence_Arousal`, `Word_frequencies`, and descriptor SVG maps.

## Required packages
`dplyr`, `readxl`, `tidyr`, `ggplot2`, `Hmisc`, `readr`, `quanteda`, `quanteda.textplots`, `quanteda.textstats`, `stringr`, `svglite`, `ggdark`, `ggthemes`, `scales`.

## Usage
1. Run `analysis_IASATposter.R` (requires `Processed Data/touch_data.txt`).
2. Run `descriptor_maps.R` to aggregate word-frequency plots into descriptor maps.
3. Check generated files in `Processed Data/` and `IASAT_poster_Sarah/Figures/`.