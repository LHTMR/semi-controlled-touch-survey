
# Analysis Record

### Aim

- Determine what are the most (and least) representative words for each video
- Compare different groups of videos per touch parameters (light vs. strong; finger vs. hand; 3cm/s vs. 9cm/s vs. 18 cm/s) and see if the calculated statistic can separate words in a way that appears to reflect the difference in modalities.
- Specific focus on `Emotional_touch` to determine whether some touches are most associated with specific social relations

### Command line

Executed from the root of the directory:

```bash
for j in $(seq 24) "3 11 7 19 17 23 9 1 21 5 13 15" "16 22 14 12 4 2 10 8 24 6 18 20" "16 14 3 4 2 17 6 18 1 5 13 15" "22 11 12 7 19 10 8 24 23 9 20 21" "16 22 14 19 17 24 18 23 20 21 13 15" "3 11 12 7 4 2 10 8 6 9 1 5" "14 7 19 2 8 20 1 13" "16 22 3 4 10 9 21 15" "11 12 17 24 6 18 23 5"; do python3 Analysis/Scripts/geometric_association_test.py --input "Processed Data/touch_data_fixed.psv.txt" --transformation-dict Analysis/All_words_by_frequency/word_grouping_dict_edited.json --plot-dir Analysis/Most_representative_words/Emotional_touch/ --results-dir Analysis/Most_representative_words/Emotional_touch/ --min-freq 2 --show-words 20 --text-columns Emotional_touch --plots "betabinom_hellinger" --target-video $j ; done
```


Data aggregation:

```bash
python3 Analysis/Scripts/aggregate_csv_data.py --input-dir Analysis/Most_representative_words/Emotional_touch
```


Data cleanup:

```bash
rm Analysis/Most_representative_words/Emotional_touch/hellinger_distance_between_betabinomial_distributions_video_*.csv
```


### Execution logs

```text
Looks like you are using a tranform that doesn't support FancyArrowPatch, using ax.annotate instead. The arrows might strike through texts. Increasing shrinkA in arrowprops might help.
============================================================
Geometric Association Test
============================================================
Input file: Processed Data/touch_data_fixed.psv.txt
Target video(s): [1]
Text columns: ['Emotional_touch']
Minimum frequency threshold: 2
Temperature gamma: 20.0
Show N words: 20
Plots to generate: ['betabinom_hellinger']
Figure dimensions: 10.0 x 8.5 inches
Plot directory: Analysis/Most_representative_words/Emotional_touch/
Results directory for CSV files: Analysis/Most_representative_words/Emotional_touch/
Transformation dictionary: Analysis/All_words_by_frequency/word_grouping_dict_edited.json
============================================================
Loading transformation dictionary from Analysis/All_words_by_frequency/word_grouping_dict_edited.json...
  Loaded transformation dictionary with 5378 mappings
Loading data from Processed Data/touch_data_fixed.psv.txt...
Loaded 2837 rows
Setting up advanced text processing...
Loading spaCy model for enhanced negation detection...
✓ Loaded spaCy model: en_core_web_md
Concatenating text columns...
Using text columns: ['Emotional_touch']
Preprocessing text for each participant (this may take a moment)...
Grouping word variations...
Using provided transformation dictionary...
Applying groups and converting to presence/absence per participant...
Calculating geometric and probabilistic metrics...

After filtering (freq >= 2): 338 words
Created results directory: Analysis/Most_representative_words/Emotional_touch/

Generating 1 plot(s)...
  CSV results saved to: Analysis/Most_representative_words/Emotional_touch/hellinger_distance_between_betabinomial_distributions_video_1.csv
10 [ 0.6387246  -0.02982693]
17 [-0.02069679 -0.73385271]
  Plot saved to: Analysis/Most_representative_words/Emotional_touch/hellinger_distance_between_betabinomial_distributions_video_1.pdf

Analysis complete!
Looks like you are using a tranform that doesn't support FancyArrowPatch, using ax.annotate instead. The arrows might strike through texts. Increasing shrinkA in arrowprops might help.
============================================================
Geometric Association Test
============================================================
Input file: Processed Data/touch_data_fixed.psv.txt
Target video(s): [2]
Text columns: ['Emotional_touch']
Minimum frequency threshold: 2
Temperature gamma: 20.0
Show N words: 20
Plots to generate: ['betabinom_hellinger']
Figure dimensions: 10.0 x 8.5 inches
Plot directory: Analysis/Most_representative_words/Emotional_touch/
Results directory for CSV files: Analysis/Most_representative_words/Emotional_touch/
Transformation dictionary: Analysis/All_words_by_frequency/word_grouping_dict_edited.json
============================================================
Loading transformation dictionary from Analysis/All_words_by_frequency/word_grouping_dict_edited.json...
  Loaded transformation dictionary with 5378 mappings
Loading data from Processed Data/touch_data_fixed.psv.txt...
Loaded 2837 rows
Setting up advanced text processing...
Loading spaCy model for enhanced negation detection...
✓ Loaded spaCy model: en_core_web_md
Concatenating text columns...
Using text columns: ['Emotional_touch']
Preprocessing text for each participant (this may take a moment)...
Grouping word variations...
Using provided transformation dictionary...
Applying groups and converting to presence/absence per participant...
Calculating geometric and probabilistic metrics...

After filtering (freq >= 2): 338 words
Created results directory: Analysis/Most_representative_words/Emotional_touch/

Generating 1 plot(s)...
  CSV results saved to: Analysis/Most_representative_words/Emotional_touch/hellinger_distance_between_betabinomial_distributions_video_2.csv
  Plot saved to: Analysis/Most_representative_words/Emotional_touch/hellinger_distance_between_betabinomial_distributions_video_2.pdf

Analysis complete!
Looks like you are using a tranform that doesn't support FancyArrowPatch, using ax.annotate instead. The arrows might strike through texts. Increasing shrinkA in arrowprops might help.
============================================================
Geometric Association Test
============================================================
Input file: Processed Data/touch_data_fixed.psv.txt
Target video(s): [3]
Text columns: ['Emotional_touch']
Minimum frequency threshold: 2
Temperature gamma: 20.0
Show N words: 20
Plots to generate: ['betabinom_hellinger']
Figure dimensions: 10.0 x 8.5 inches
Plot directory: Analysis/Most_representative_words/Emotional_touch/
Results directory for CSV files: Analysis/Most_representative_words/Emotional_touch/
Transformation dictionary: Analysis/All_words_by_frequency/word_grouping_dict_edited.json
============================================================
Loading transformation dictionary from Analysis/All_words_by_frequency/word_grouping_dict_edited.json...
  Loaded transformation dictionary with 5378 mappings
Loading data from Processed Data/touch_data_fixed.psv.txt...
Loaded 2837 rows
Setting up advanced text processing...
Loading spaCy model for enhanced negation detection...
✓ Loaded spaCy model: en_core_web_md
Concatenating text columns...
Using text columns: ['Emotional_touch']
Preprocessing text for each participant (this may take a moment)...
Grouping word variations...
Using provided transformation dictionary...
Applying groups and converting to presence/absence per participant...
Calculating geometric and probabilistic metrics...

After filtering (freq >= 2): 338 words
Created results directory: Analysis/Most_representative_words/Emotional_touch/

Generating 1 plot(s)...
  CSV results saved to: Analysis/Most_representative_words/Emotional_touch/hellinger_distance_between_betabinomial_distributions_video_3.csv
11 [-0.80765913 -0.18260707]
14 [0.64562099 0.75544664]
  Plot saved to: Analysis/Most_representative_words/Emotional_touch/hellinger_distance_between_betabinomial_distributions_video_3.pdf

Analysis complete!
Looks like you are using a tranform that doesn't support FancyArrowPatch, using ax.annotate instead. The arrows might strike through texts. Increasing shrinkA in arrowprops might help.
============================================================
Geometric Association Test
============================================================
Input file: Processed Data/touch_data_fixed.psv.txt
Target video(s): [4]
Text columns: ['Emotional_touch']
Minimum frequency threshold: 2
Temperature gamma: 20.0
Show N words: 20
Plots to generate: ['betabinom_hellinger']
Figure dimensions: 10.0 x 8.5 inches
Plot directory: Analysis/Most_representative_words/Emotional_touch/
Results directory for CSV files: Analysis/Most_representative_words/Emotional_touch/
Transformation dictionary: Analysis/All_words_by_frequency/word_grouping_dict_edited.json
============================================================
Loading transformation dictionary from Analysis/All_words_by_frequency/word_grouping_dict_edited.json...
  Loaded transformation dictionary with 5378 mappings
Loading data from Processed Data/touch_data_fixed.psv.txt...
Loaded 2837 rows
Setting up advanced text processing...
Loading spaCy model for enhanced negation detection...
✓ Loaded spaCy model: en_core_web_md
Concatenating text columns...
Using text columns: ['Emotional_touch']
Preprocessing text for each participant (this may take a moment)...
Grouping word variations...
Using provided transformation dictionary...
Applying groups and converting to presence/absence per participant...
Calculating geometric and probabilistic metrics...

After filtering (freq >= 2): 338 words
Created results directory: Analysis/Most_representative_words/Emotional_touch/

Generating 1 plot(s)...
  CSV results saved to: Analysis/Most_representative_words/Emotional_touch/hellinger_distance_between_betabinomial_distributions_video_4.csv
  Plot saved to: Analysis/Most_representative_words/Emotional_touch/hellinger_distance_between_betabinomial_distributions_video_4.pdf

Analysis complete!
Looks like you are using a tranform that doesn't support FancyArrowPatch, using ax.annotate instead. The arrows might strike through texts. Increasing shrinkA in arrowprops might help.
============================================================
Geometric Association Test
============================================================
Input file: Processed Data/touch_data_fixed.psv.txt
Target video(s): [5]
Text columns: ['Emotional_touch']
Minimum frequency threshold: 2
Temperature gamma: 20.0
Show N words: 20
Plots to generate: ['betabinom_hellinger']
Figure dimensions: 10.0 x 8.5 inches
Plot directory: Analysis/Most_representative_words/Emotional_touch/
Results directory for CSV files: Analysis/Most_representative_words/Emotional_touch/
Transformation dictionary: Analysis/All_words_by_frequency/word_grouping_dict_edited.json
============================================================
Loading transformation dictionary from Analysis/All_words_by_frequency/word_grouping_dict_edited.json...
  Loaded transformation dictionary with 5378 mappings
Loading data from Processed Data/touch_data_fixed.psv.txt...
Loaded 2837 rows
Setting up advanced text processing...
Loading spaCy model for enhanced negation detection...
✓ Loaded spaCy model: en_core_web_md
Concatenating text columns...
Using text columns: ['Emotional_touch']
Preprocessing text for each participant (this may take a moment)...
Grouping word variations...
Using provided transformation dictionary...
Applying groups and converting to presence/absence per participant...
Calculating geometric and probabilistic metrics...

After filtering (freq >= 2): 338 words
Created results directory: Analysis/Most_representative_words/Emotional_touch/

Generating 1 plot(s)...
  CSV results saved to: Analysis/Most_representative_words/Emotional_touch/hellinger_distance_between_betabinomial_distributions_video_5.csv
  Plot saved to: Analysis/Most_representative_words/Emotional_touch/hellinger_distance_between_betabinomial_distributions_video_5.pdf

Analysis complete!
Looks like you are using a tranform that doesn't support FancyArrowPatch, using ax.annotate instead. The arrows might strike through texts. Increasing shrinkA in arrowprops might help.
============================================================
Geometric Association Test
============================================================
Input file: Processed Data/touch_data_fixed.psv.txt
Target video(s): [6]
Text columns: ['Emotional_touch']
Minimum frequency threshold: 2
Temperature gamma: 20.0
Show N words: 20
Plots to generate: ['betabinom_hellinger']
Figure dimensions: 10.0 x 8.5 inches
Plot directory: Analysis/Most_representative_words/Emotional_touch/
Results directory for CSV files: Analysis/Most_representative_words/Emotional_touch/
Transformation dictionary: Analysis/All_words_by_frequency/word_grouping_dict_edited.json
============================================================
Loading transformation dictionary from Analysis/All_words_by_frequency/word_grouping_dict_edited.json...
  Loaded transformation dictionary with 5378 mappings
Loading data from Processed Data/touch_data_fixed.psv.txt...
Loaded 2837 rows
Setting up advanced text processing...
Loading spaCy model for enhanced negation detection...
✓ Loaded spaCy model: en_core_web_md
Concatenating text columns...
Using text columns: ['Emotional_touch']
Preprocessing text for each participant (this may take a moment)...
Grouping word variations...
Using provided transformation dictionary...
Applying groups and converting to presence/absence per participant...
Calculating geometric and probabilistic metrics...

After filtering (freq >= 2): 338 words
Created results directory: Analysis/Most_representative_words/Emotional_touch/

Generating 1 plot(s)...
  CSV results saved to: Analysis/Most_representative_words/Emotional_touch/hellinger_distance_between_betabinomial_distributions_video_6.csv
  Plot saved to: Analysis/Most_representative_words/Emotional_touch/hellinger_distance_between_betabinomial_distributions_video_6.pdf

Analysis complete!
Looks like you are using a tranform that doesn't support FancyArrowPatch, using ax.annotate instead. The arrows might strike through texts. Increasing shrinkA in arrowprops might help.
============================================================
Geometric Association Test
============================================================
Input file: Processed Data/touch_data_fixed.psv.txt
Target video(s): [7]
Text columns: ['Emotional_touch']
Minimum frequency threshold: 2
Temperature gamma: 20.0
Show N words: 20
Plots to generate: ['betabinom_hellinger']
Figure dimensions: 10.0 x 8.5 inches
Plot directory: Analysis/Most_representative_words/Emotional_touch/
Results directory for CSV files: Analysis/Most_representative_words/Emotional_touch/
Transformation dictionary: Analysis/All_words_by_frequency/word_grouping_dict_edited.json
============================================================
Loading transformation dictionary from Analysis/All_words_by_frequency/word_grouping_dict_edited.json...
  Loaded transformation dictionary with 5378 mappings
Loading data from Processed Data/touch_data_fixed.psv.txt...
Loaded 2837 rows
Setting up advanced text processing...
Loading spaCy model for enhanced negation detection...
✓ Loaded spaCy model: en_core_web_md
Concatenating text columns...
Using text columns: ['Emotional_touch']
Preprocessing text for each participant (this may take a moment)...
Grouping word variations...
Using provided transformation dictionary...
Applying groups and converting to presence/absence per participant...
Calculating geometric and probabilistic metrics...

After filtering (freq >= 2): 338 words
Created results directory: Analysis/Most_representative_words/Emotional_touch/

Generating 1 plot(s)...
  CSV results saved to: Analysis/Most_representative_words/Emotional_touch/hellinger_distance_between_betabinomial_distributions_video_7.csv
  Plot saved to: Analysis/Most_representative_words/Emotional_touch/hellinger_distance_between_betabinomial_distributions_video_7.pdf

Analysis complete!
Looks like you are using a tranform that doesn't support FancyArrowPatch, using ax.annotate instead. The arrows might strike through texts. Increasing shrinkA in arrowprops might help.
============================================================
Geometric Association Test
============================================================
Input file: Processed Data/touch_data_fixed.psv.txt
Target video(s): [8]
Text columns: ['Emotional_touch']
Minimum frequency threshold: 2
Temperature gamma: 20.0
Show N words: 20
Plots to generate: ['betabinom_hellinger']
Figure dimensions: 10.0 x 8.5 inches
Plot directory: Analysis/Most_representative_words/Emotional_touch/
Results directory for CSV files: Analysis/Most_representative_words/Emotional_touch/
Transformation dictionary: Analysis/All_words_by_frequency/word_grouping_dict_edited.json
============================================================
Loading transformation dictionary from Analysis/All_words_by_frequency/word_grouping_dict_edited.json...
  Loaded transformation dictionary with 5378 mappings
Loading data from Processed Data/touch_data_fixed.psv.txt...
Loaded 2837 rows
Setting up advanced text processing...
Loading spaCy model for enhanced negation detection...
✓ Loaded spaCy model: en_core_web_md
Concatenating text columns...
Using text columns: ['Emotional_touch']
Preprocessing text for each participant (this may take a moment)...
Grouping word variations...
Using provided transformation dictionary...
Applying groups and converting to presence/absence per participant...
Calculating geometric and probabilistic metrics...

After filtering (freq >= 2): 338 words
Created results directory: Analysis/Most_representative_words/Emotional_touch/

Generating 1 plot(s)...
  CSV results saved to: Analysis/Most_representative_words/Emotional_touch/hellinger_distance_between_betabinomial_distributions_video_8.csv
4 [-0.4139104   0.13392793]
8 [-0.77835885 -0.95801928]
  Plot saved to: Analysis/Most_representative_words/Emotional_touch/hellinger_distance_between_betabinomial_distributions_video_8.pdf

Analysis complete!
Looks like you are using a tranform that doesn't support FancyArrowPatch, using ax.annotate instead. The arrows might strike through texts. Increasing shrinkA in arrowprops might help.
============================================================
Geometric Association Test
============================================================
Input file: Processed Data/touch_data_fixed.psv.txt
Target video(s): [9]
Text columns: ['Emotional_touch']
Minimum frequency threshold: 2
Temperature gamma: 20.0
Show N words: 20
Plots to generate: ['betabinom_hellinger']
Figure dimensions: 10.0 x 8.5 inches
Plot directory: Analysis/Most_representative_words/Emotional_touch/
Results directory for CSV files: Analysis/Most_representative_words/Emotional_touch/
Transformation dictionary: Analysis/All_words_by_frequency/word_grouping_dict_edited.json
============================================================
Loading transformation dictionary from Analysis/All_words_by_frequency/word_grouping_dict_edited.json...
  Loaded transformation dictionary with 5378 mappings
Loading data from Processed Data/touch_data_fixed.psv.txt...
Loaded 2837 rows
Setting up advanced text processing...
Loading spaCy model for enhanced negation detection...
✓ Loaded spaCy model: en_core_web_md
Concatenating text columns...
Using text columns: ['Emotional_touch']
Preprocessing text for each participant (this may take a moment)...
Grouping word variations...
Using provided transformation dictionary...
Applying groups and converting to presence/absence per participant...
Calculating geometric and probabilistic metrics...

After filtering (freq >= 2): 338 words
Created results directory: Analysis/Most_representative_words/Emotional_touch/

Generating 1 plot(s)...
  CSV results saved to: Analysis/Most_representative_words/Emotional_touch/hellinger_distance_between_betabinomial_distributions_video_9.csv
15 [0.32631965 0.24906123]
18 [ 0.34054085 -0.25389868]
  Plot saved to: Analysis/Most_representative_words/Emotional_touch/hellinger_distance_between_betabinomial_distributions_video_9.pdf

Analysis complete!
Looks like you are using a tranform that doesn't support FancyArrowPatch, using ax.annotate instead. The arrows might strike through texts. Increasing shrinkA in arrowprops might help.
============================================================
Geometric Association Test
============================================================
Input file: Processed Data/touch_data_fixed.psv.txt
Target video(s): [10]
Text columns: ['Emotional_touch']
Minimum frequency threshold: 2
Temperature gamma: 20.0
Show N words: 20
Plots to generate: ['betabinom_hellinger']
Figure dimensions: 10.0 x 8.5 inches
Plot directory: Analysis/Most_representative_words/Emotional_touch/
Results directory for CSV files: Analysis/Most_representative_words/Emotional_touch/
Transformation dictionary: Analysis/All_words_by_frequency/word_grouping_dict_edited.json
============================================================
Loading transformation dictionary from Analysis/All_words_by_frequency/word_grouping_dict_edited.json...
  Loaded transformation dictionary with 5378 mappings
Loading data from Processed Data/touch_data_fixed.psv.txt...
Loaded 2837 rows
Setting up advanced text processing...
Loading spaCy model for enhanced negation detection...
✓ Loaded spaCy model: en_core_web_md
Concatenating text columns...
Using text columns: ['Emotional_touch']
Preprocessing text for each participant (this may take a moment)...
Grouping word variations...
Using provided transformation dictionary...
Applying groups and converting to presence/absence per participant...
Calculating geometric and probabilistic metrics...

After filtering (freq >= 2): 338 words
Created results directory: Analysis/Most_representative_words/Emotional_touch/

Generating 1 plot(s)...
  CSV results saved to: Analysis/Most_representative_words/Emotional_touch/hellinger_distance_between_betabinomial_distributions_video_10.csv
9 [0.72264471 0.30398824]
10 [ 0.9278889  -0.73872341]
  Plot saved to: Analysis/Most_representative_words/Emotional_touch/hellinger_distance_between_betabinomial_distributions_video_10.pdf

Analysis complete!
Looks like you are using a tranform that doesn't support FancyArrowPatch, using ax.annotate instead. The arrows might strike through texts. Increasing shrinkA in arrowprops might help.
============================================================
Geometric Association Test
============================================================
Input file: Processed Data/touch_data_fixed.psv.txt
Target video(s): [11]
Text columns: ['Emotional_touch']
Minimum frequency threshold: 2
Temperature gamma: 20.0
Show N words: 20
Plots to generate: ['betabinom_hellinger']
Figure dimensions: 10.0 x 8.5 inches
Plot directory: Analysis/Most_representative_words/Emotional_touch/
Results directory for CSV files: Analysis/Most_representative_words/Emotional_touch/
Transformation dictionary: Analysis/All_words_by_frequency/word_grouping_dict_edited.json
============================================================
Loading transformation dictionary from Analysis/All_words_by_frequency/word_grouping_dict_edited.json...
  Loaded transformation dictionary with 5378 mappings
Loading data from Processed Data/touch_data_fixed.psv.txt...
Loaded 2837 rows
Setting up advanced text processing...
Loading spaCy model for enhanced negation detection...
✓ Loaded spaCy model: en_core_web_md
Concatenating text columns...
Using text columns: ['Emotional_touch']
Preprocessing text for each participant (this may take a moment)...
Grouping word variations...
Using provided transformation dictionary...
Applying groups and converting to presence/absence per participant...
Calculating geometric and probabilistic metrics...

After filtering (freq >= 2): 338 words
Created results directory: Analysis/Most_representative_words/Emotional_touch/

Generating 1 plot(s)...
  CSV results saved to: Analysis/Most_representative_words/Emotional_touch/hellinger_distance_between_betabinomial_distributions_video_11.csv
10 [0.09428801 0.31125732]
12 [-0.48654046 -0.54311478]
17 [0.64928289 0.39942729]
18 [-0.0880323 -0.5991684]
  Plot saved to: Analysis/Most_representative_words/Emotional_touch/hellinger_distance_between_betabinomial_distributions_video_11.pdf

Analysis complete!
Looks like you are using a tranform that doesn't support FancyArrowPatch, using ax.annotate instead. The arrows might strike through texts. Increasing shrinkA in arrowprops might help.
============================================================
Geometric Association Test
============================================================
Input file: Processed Data/touch_data_fixed.psv.txt
Target video(s): [12]
Text columns: ['Emotional_touch']
Minimum frequency threshold: 2
Temperature gamma: 20.0
Show N words: 20
Plots to generate: ['betabinom_hellinger']
Figure dimensions: 10.0 x 8.5 inches
Plot directory: Analysis/Most_representative_words/Emotional_touch/
Results directory for CSV files: Analysis/Most_representative_words/Emotional_touch/
Transformation dictionary: Analysis/All_words_by_frequency/word_grouping_dict_edited.json
============================================================
Loading transformation dictionary from Analysis/All_words_by_frequency/word_grouping_dict_edited.json...
  Loaded transformation dictionary with 5378 mappings
Loading data from Processed Data/touch_data_fixed.psv.txt...
Loaded 2837 rows
Setting up advanced text processing...
Loading spaCy model for enhanced negation detection...
✓ Loaded spaCy model: en_core_web_md
Concatenating text columns...
Using text columns: ['Emotional_touch']
Preprocessing text for each participant (this may take a moment)...
Grouping word variations...
Using provided transformation dictionary...
Applying groups and converting to presence/absence per participant...
Calculating geometric and probabilistic metrics...

After filtering (freq >= 2): 338 words
Created results directory: Analysis/Most_representative_words/Emotional_touch/

Generating 1 plot(s)...
  CSV results saved to: Analysis/Most_representative_words/Emotional_touch/hellinger_distance_between_betabinomial_distributions_video_12.csv
10 [ 0.52303679 -0.69235844]
15 [0.25840196 0.71546091]
6 [ 0.12886882 -0.93170714]
11 [-0.03313594  0.98483256]
  Plot saved to: Analysis/Most_representative_words/Emotional_touch/hellinger_distance_between_betabinomial_distributions_video_12.pdf

Analysis complete!
Looks like you are using a tranform that doesn't support FancyArrowPatch, using ax.annotate instead. The arrows might strike through texts. Increasing shrinkA in arrowprops might help.
============================================================
Geometric Association Test
============================================================
Input file: Processed Data/touch_data_fixed.psv.txt
Target video(s): [13]
Text columns: ['Emotional_touch']
Minimum frequency threshold: 2
Temperature gamma: 20.0
Show N words: 20
Plots to generate: ['betabinom_hellinger']
Figure dimensions: 10.0 x 8.5 inches
Plot directory: Analysis/Most_representative_words/Emotional_touch/
Results directory for CSV files: Analysis/Most_representative_words/Emotional_touch/
Transformation dictionary: Analysis/All_words_by_frequency/word_grouping_dict_edited.json
============================================================
Loading transformation dictionary from Analysis/All_words_by_frequency/word_grouping_dict_edited.json...
  Loaded transformation dictionary with 5378 mappings
Loading data from Processed Data/touch_data_fixed.psv.txt...
Loaded 2837 rows
Setting up advanced text processing...
Loading spaCy model for enhanced negation detection...
✓ Loaded spaCy model: en_core_web_md
Concatenating text columns...
Using text columns: ['Emotional_touch']
Preprocessing text for each participant (this may take a moment)...
Grouping word variations...
Using provided transformation dictionary...
Applying groups and converting to presence/absence per participant...
Calculating geometric and probabilistic metrics...

After filtering (freq >= 2): 338 words
Created results directory: Analysis/Most_representative_words/Emotional_touch/

Generating 1 plot(s)...
  CSV results saved to: Analysis/Most_representative_words/Emotional_touch/hellinger_distance_between_betabinomial_distributions_video_13.csv
  Plot saved to: Analysis/Most_representative_words/Emotional_touch/hellinger_distance_between_betabinomial_distributions_video_13.pdf

Analysis complete!
Looks like you are using a tranform that doesn't support FancyArrowPatch, using ax.annotate instead. The arrows might strike through texts. Increasing shrinkA in arrowprops might help.
============================================================
Geometric Association Test
============================================================
Input file: Processed Data/touch_data_fixed.psv.txt
Target video(s): [14]
Text columns: ['Emotional_touch']
Minimum frequency threshold: 2
Temperature gamma: 20.0
Show N words: 20
Plots to generate: ['betabinom_hellinger']
Figure dimensions: 10.0 x 8.5 inches
Plot directory: Analysis/Most_representative_words/Emotional_touch/
Results directory for CSV files: Analysis/Most_representative_words/Emotional_touch/
Transformation dictionary: Analysis/All_words_by_frequency/word_grouping_dict_edited.json
============================================================
Loading transformation dictionary from Analysis/All_words_by_frequency/word_grouping_dict_edited.json...
  Loaded transformation dictionary with 5378 mappings
Loading data from Processed Data/touch_data_fixed.psv.txt...
Loaded 2837 rows
Setting up advanced text processing...
Loading spaCy model for enhanced negation detection...
✓ Loaded spaCy model: en_core_web_md
Concatenating text columns...
Using text columns: ['Emotional_touch']
Preprocessing text for each participant (this may take a moment)...
Grouping word variations...
Using provided transformation dictionary...
Applying groups and converting to presence/absence per participant...
Calculating geometric and probabilistic metrics...

After filtering (freq >= 2): 338 words
Created results directory: Analysis/Most_representative_words/Emotional_touch/

Generating 1 plot(s)...
  CSV results saved to: Analysis/Most_representative_words/Emotional_touch/hellinger_distance_between_betabinomial_distributions_video_14.csv
7 [ 0.89251861 -0.89908183]
8 [0.65490505 0.46184707]
  Plot saved to: Analysis/Most_representative_words/Emotional_touch/hellinger_distance_between_betabinomial_distributions_video_14.pdf

Analysis complete!
Looks like you are using a tranform that doesn't support FancyArrowPatch, using ax.annotate instead. The arrows might strike through texts. Increasing shrinkA in arrowprops might help.
============================================================
Geometric Association Test
============================================================
Input file: Processed Data/touch_data_fixed.psv.txt
Target video(s): [15]
Text columns: ['Emotional_touch']
Minimum frequency threshold: 2
Temperature gamma: 20.0
Show N words: 20
Plots to generate: ['betabinom_hellinger']
Figure dimensions: 10.0 x 8.5 inches
Plot directory: Analysis/Most_representative_words/Emotional_touch/
Results directory for CSV files: Analysis/Most_representative_words/Emotional_touch/
Transformation dictionary: Analysis/All_words_by_frequency/word_grouping_dict_edited.json
============================================================
Loading transformation dictionary from Analysis/All_words_by_frequency/word_grouping_dict_edited.json...
  Loaded transformation dictionary with 5378 mappings
Loading data from Processed Data/touch_data_fixed.psv.txt...
Loaded 2837 rows
Setting up advanced text processing...
Loading spaCy model for enhanced negation detection...
✓ Loaded spaCy model: en_core_web_md
Concatenating text columns...
Using text columns: ['Emotional_touch']
Preprocessing text for each participant (this may take a moment)...
Grouping word variations...
Using provided transformation dictionary...
Applying groups and converting to presence/absence per participant...
Calculating geometric and probabilistic metrics...

After filtering (freq >= 2): 338 words
Created results directory: Analysis/Most_representative_words/Emotional_touch/

Generating 1 plot(s)...
  CSV results saved to: Analysis/Most_representative_words/Emotional_touch/hellinger_distance_between_betabinomial_distributions_video_15.csv
7 [-0.42129879  0.81454685]
9 [0.26028816 0.06229296]
  Plot saved to: Analysis/Most_representative_words/Emotional_touch/hellinger_distance_between_betabinomial_distributions_video_15.pdf

Analysis complete!
Looks like you are using a tranform that doesn't support FancyArrowPatch, using ax.annotate instead. The arrows might strike through texts. Increasing shrinkA in arrowprops might help.
============================================================
Geometric Association Test
============================================================
Input file: Processed Data/touch_data_fixed.psv.txt
Target video(s): [16]
Text columns: ['Emotional_touch']
Minimum frequency threshold: 2
Temperature gamma: 20.0
Show N words: 20
Plots to generate: ['betabinom_hellinger']
Figure dimensions: 10.0 x 8.5 inches
Plot directory: Analysis/Most_representative_words/Emotional_touch/
Results directory for CSV files: Analysis/Most_representative_words/Emotional_touch/
Transformation dictionary: Analysis/All_words_by_frequency/word_grouping_dict_edited.json
============================================================
Loading transformation dictionary from Analysis/All_words_by_frequency/word_grouping_dict_edited.json...
  Loaded transformation dictionary with 5378 mappings
Loading data from Processed Data/touch_data_fixed.psv.txt...
Loaded 2837 rows
Setting up advanced text processing...
Loading spaCy model for enhanced negation detection...
✓ Loaded spaCy model: en_core_web_md
Concatenating text columns...
Using text columns: ['Emotional_touch']
Preprocessing text for each participant (this may take a moment)...
Grouping word variations...
Using provided transformation dictionary...
Applying groups and converting to presence/absence per participant...
Calculating geometric and probabilistic metrics...

After filtering (freq >= 2): 338 words
Created results directory: Analysis/Most_representative_words/Emotional_touch/

Generating 1 plot(s)...
  CSV results saved to: Analysis/Most_representative_words/Emotional_touch/hellinger_distance_between_betabinomial_distributions_video_16.csv
13 [ 0.36929757 -0.4358549 ]
14 [-0.83853961  0.87870337]
8 [ 0.44230284 -0.56477826]
17 [ 0.41782789 -0.0818939 ]
  Plot saved to: Analysis/Most_representative_words/Emotional_touch/hellinger_distance_between_betabinomial_distributions_video_16.pdf

Analysis complete!
Looks like you are using a tranform that doesn't support FancyArrowPatch, using ax.annotate instead. The arrows might strike through texts. Increasing shrinkA in arrowprops might help.
============================================================
Geometric Association Test
============================================================
Input file: Processed Data/touch_data_fixed.psv.txt
Target video(s): [17]
Text columns: ['Emotional_touch']
Minimum frequency threshold: 2
Temperature gamma: 20.0
Show N words: 20
Plots to generate: ['betabinom_hellinger']
Figure dimensions: 10.0 x 8.5 inches
Plot directory: Analysis/Most_representative_words/Emotional_touch/
Results directory for CSV files: Analysis/Most_representative_words/Emotional_touch/
Transformation dictionary: Analysis/All_words_by_frequency/word_grouping_dict_edited.json
============================================================
Loading transformation dictionary from Analysis/All_words_by_frequency/word_grouping_dict_edited.json...
  Loaded transformation dictionary with 5378 mappings
Loading data from Processed Data/touch_data_fixed.psv.txt...
Loaded 2837 rows
Setting up advanced text processing...
Loading spaCy model for enhanced negation detection...
✓ Loaded spaCy model: en_core_web_md
Concatenating text columns...
Using text columns: ['Emotional_touch']
Preprocessing text for each participant (this may take a moment)...
Grouping word variations...
Using provided transformation dictionary...
Applying groups and converting to presence/absence per participant...
Calculating geometric and probabilistic metrics...

After filtering (freq >= 2): 338 words
Created results directory: Analysis/Most_representative_words/Emotional_touch/

Generating 1 plot(s)...
  CSV results saved to: Analysis/Most_representative_words/Emotional_touch/hellinger_distance_between_betabinomial_distributions_video_17.csv
8 [ 0.6371209  -0.19072501]
10 [-0.8804022   0.34970542]
14 [ 0.17076235 -0.67606513]
15 [-0.03494191 -0.10559725]
14 [0.23156816 0.48060746]
16 [-0.44010546  0.64548999]
  Plot saved to: Analysis/Most_representative_words/Emotional_touch/hellinger_distance_between_betabinomial_distributions_video_17.pdf

Analysis complete!
Looks like you are using a tranform that doesn't support FancyArrowPatch, using ax.annotate instead. The arrows might strike through texts. Increasing shrinkA in arrowprops might help.
============================================================
Geometric Association Test
============================================================
Input file: Processed Data/touch_data_fixed.psv.txt
Target video(s): [18]
Text columns: ['Emotional_touch']
Minimum frequency threshold: 2
Temperature gamma: 20.0
Show N words: 20
Plots to generate: ['betabinom_hellinger']
Figure dimensions: 10.0 x 8.5 inches
Plot directory: Analysis/Most_representative_words/Emotional_touch/
Results directory for CSV files: Analysis/Most_representative_words/Emotional_touch/
Transformation dictionary: Analysis/All_words_by_frequency/word_grouping_dict_edited.json
============================================================
Loading transformation dictionary from Analysis/All_words_by_frequency/word_grouping_dict_edited.json...
  Loaded transformation dictionary with 5378 mappings
Loading data from Processed Data/touch_data_fixed.psv.txt...
Loaded 2837 rows
Setting up advanced text processing...
Loading spaCy model for enhanced negation detection...
✓ Loaded spaCy model: en_core_web_md
Concatenating text columns...
Using text columns: ['Emotional_touch']
Preprocessing text for each participant (this may take a moment)...
Grouping word variations...
Using provided transformation dictionary...
Applying groups and converting to presence/absence per participant...
Calculating geometric and probabilistic metrics...

After filtering (freq >= 2): 338 words
Created results directory: Analysis/Most_representative_words/Emotional_touch/

Generating 1 plot(s)...
  CSV results saved to: Analysis/Most_representative_words/Emotional_touch/hellinger_distance_between_betabinomial_distributions_video_18.csv
  Plot saved to: Analysis/Most_representative_words/Emotional_touch/hellinger_distance_between_betabinomial_distributions_video_18.pdf

Analysis complete!
Looks like you are using a tranform that doesn't support FancyArrowPatch, using ax.annotate instead. The arrows might strike through texts. Increasing shrinkA in arrowprops might help.
============================================================
Geometric Association Test
============================================================
Input file: Processed Data/touch_data_fixed.psv.txt
Target video(s): [19]
Text columns: ['Emotional_touch']
Minimum frequency threshold: 2
Temperature gamma: 20.0
Show N words: 20
Plots to generate: ['betabinom_hellinger']
Figure dimensions: 10.0 x 8.5 inches
Plot directory: Analysis/Most_representative_words/Emotional_touch/
Results directory for CSV files: Analysis/Most_representative_words/Emotional_touch/
Transformation dictionary: Analysis/All_words_by_frequency/word_grouping_dict_edited.json
============================================================
Loading transformation dictionary from Analysis/All_words_by_frequency/word_grouping_dict_edited.json...
  Loaded transformation dictionary with 5378 mappings
Loading data from Processed Data/touch_data_fixed.psv.txt...
Loaded 2837 rows
Setting up advanced text processing...
Loading spaCy model for enhanced negation detection...
✓ Loaded spaCy model: en_core_web_md
Concatenating text columns...
Using text columns: ['Emotional_touch']
Preprocessing text for each participant (this may take a moment)...
Grouping word variations...
Using provided transformation dictionary...
Applying groups and converting to presence/absence per participant...
Calculating geometric and probabilistic metrics...

After filtering (freq >= 2): 338 words
Created results directory: Analysis/Most_representative_words/Emotional_touch/

Generating 1 plot(s)...
  CSV results saved to: Analysis/Most_representative_words/Emotional_touch/hellinger_distance_between_betabinomial_distributions_video_19.csv
  Plot saved to: Analysis/Most_representative_words/Emotional_touch/hellinger_distance_between_betabinomial_distributions_video_19.pdf

Analysis complete!
Looks like you are using a tranform that doesn't support FancyArrowPatch, using ax.annotate instead. The arrows might strike through texts. Increasing shrinkA in arrowprops might help.
============================================================
Geometric Association Test
============================================================
Input file: Processed Data/touch_data_fixed.psv.txt
Target video(s): [20]
Text columns: ['Emotional_touch']
Minimum frequency threshold: 2
Temperature gamma: 20.0
Show N words: 20
Plots to generate: ['betabinom_hellinger']
Figure dimensions: 10.0 x 8.5 inches
Plot directory: Analysis/Most_representative_words/Emotional_touch/
Results directory for CSV files: Analysis/Most_representative_words/Emotional_touch/
Transformation dictionary: Analysis/All_words_by_frequency/word_grouping_dict_edited.json
============================================================
Loading transformation dictionary from Analysis/All_words_by_frequency/word_grouping_dict_edited.json...
  Loaded transformation dictionary with 5378 mappings
Loading data from Processed Data/touch_data_fixed.psv.txt...
Loaded 2837 rows
Setting up advanced text processing...
Loading spaCy model for enhanced negation detection...
✓ Loaded spaCy model: en_core_web_md
Concatenating text columns...
Using text columns: ['Emotional_touch']
Preprocessing text for each participant (this may take a moment)...
Grouping word variations...
Using provided transformation dictionary...
Applying groups and converting to presence/absence per participant...
Calculating geometric and probabilistic metrics...

After filtering (freq >= 2): 338 words
Created results directory: Analysis/Most_representative_words/Emotional_touch/

Generating 1 plot(s)...
  CSV results saved to: Analysis/Most_representative_words/Emotional_touch/hellinger_distance_between_betabinomial_distributions_video_20.csv
  Plot saved to: Analysis/Most_representative_words/Emotional_touch/hellinger_distance_between_betabinomial_distributions_video_20.pdf

Analysis complete!
Looks like you are using a tranform that doesn't support FancyArrowPatch, using ax.annotate instead. The arrows might strike through texts. Increasing shrinkA in arrowprops might help.
============================================================
Geometric Association Test
============================================================
Input file: Processed Data/touch_data_fixed.psv.txt
Target video(s): [21]
Text columns: ['Emotional_touch']
Minimum frequency threshold: 2
Temperature gamma: 20.0
Show N words: 20
Plots to generate: ['betabinom_hellinger']
Figure dimensions: 10.0 x 8.5 inches
Plot directory: Analysis/Most_representative_words/Emotional_touch/
Results directory for CSV files: Analysis/Most_representative_words/Emotional_touch/
Transformation dictionary: Analysis/All_words_by_frequency/word_grouping_dict_edited.json
============================================================
Loading transformation dictionary from Analysis/All_words_by_frequency/word_grouping_dict_edited.json...
  Loaded transformation dictionary with 5378 mappings
Loading data from Processed Data/touch_data_fixed.psv.txt...
Loaded 2837 rows
Setting up advanced text processing...
Loading spaCy model for enhanced negation detection...
✓ Loaded spaCy model: en_core_web_md
Concatenating text columns...
Using text columns: ['Emotional_touch']
Preprocessing text for each participant (this may take a moment)...
Grouping word variations...
Using provided transformation dictionary...
Applying groups and converting to presence/absence per participant...
Calculating geometric and probabilistic metrics...

After filtering (freq >= 2): 338 words
Created results directory: Analysis/Most_representative_words/Emotional_touch/

Generating 1 plot(s)...
  CSV results saved to: Analysis/Most_representative_words/Emotional_touch/hellinger_distance_between_betabinomial_distributions_video_21.csv
  Plot saved to: Analysis/Most_representative_words/Emotional_touch/hellinger_distance_between_betabinomial_distributions_video_21.pdf

Analysis complete!
Looks like you are using a tranform that doesn't support FancyArrowPatch, using ax.annotate instead. The arrows might strike through texts. Increasing shrinkA in arrowprops might help.
============================================================
Geometric Association Test
============================================================
Input file: Processed Data/touch_data_fixed.psv.txt
Target video(s): [22]
Text columns: ['Emotional_touch']
Minimum frequency threshold: 2
Temperature gamma: 20.0
Show N words: 20
Plots to generate: ['betabinom_hellinger']
Figure dimensions: 10.0 x 8.5 inches
Plot directory: Analysis/Most_representative_words/Emotional_touch/
Results directory for CSV files: Analysis/Most_representative_words/Emotional_touch/
Transformation dictionary: Analysis/All_words_by_frequency/word_grouping_dict_edited.json
============================================================
Loading transformation dictionary from Analysis/All_words_by_frequency/word_grouping_dict_edited.json...
  Loaded transformation dictionary with 5378 mappings
Loading data from Processed Data/touch_data_fixed.psv.txt...
Loaded 2837 rows
Setting up advanced text processing...
Loading spaCy model for enhanced negation detection...
✓ Loaded spaCy model: en_core_web_md
Concatenating text columns...
Using text columns: ['Emotional_touch']
Preprocessing text for each participant (this may take a moment)...
Grouping word variations...
Using provided transformation dictionary...
Applying groups and converting to presence/absence per participant...
Calculating geometric and probabilistic metrics...

After filtering (freq >= 2): 338 words
Created results directory: Analysis/Most_representative_words/Emotional_touch/

Generating 1 plot(s)...
  CSV results saved to: Analysis/Most_representative_words/Emotional_touch/hellinger_distance_between_betabinomial_distributions_video_22.csv
  Plot saved to: Analysis/Most_representative_words/Emotional_touch/hellinger_distance_between_betabinomial_distributions_video_22.pdf

Analysis complete!
Looks like you are using a tranform that doesn't support FancyArrowPatch, using ax.annotate instead. The arrows might strike through texts. Increasing shrinkA in arrowprops might help.
============================================================
Geometric Association Test
============================================================
Input file: Processed Data/touch_data_fixed.psv.txt
Target video(s): [23]
Text columns: ['Emotional_touch']
Minimum frequency threshold: 2
Temperature gamma: 20.0
Show N words: 20
Plots to generate: ['betabinom_hellinger']
Figure dimensions: 10.0 x 8.5 inches
Plot directory: Analysis/Most_representative_words/Emotional_touch/
Results directory for CSV files: Analysis/Most_representative_words/Emotional_touch/
Transformation dictionary: Analysis/All_words_by_frequency/word_grouping_dict_edited.json
============================================================
Loading transformation dictionary from Analysis/All_words_by_frequency/word_grouping_dict_edited.json...
  Loaded transformation dictionary with 5378 mappings
Loading data from Processed Data/touch_data_fixed.psv.txt...
Loaded 2837 rows
Setting up advanced text processing...
Loading spaCy model for enhanced negation detection...
✓ Loaded spaCy model: en_core_web_md
Concatenating text columns...
Using text columns: ['Emotional_touch']
Preprocessing text for each participant (this may take a moment)...
Grouping word variations...
Using provided transformation dictionary...
Applying groups and converting to presence/absence per participant...
Calculating geometric and probabilistic metrics...

After filtering (freq >= 2): 338 words
Created results directory: Analysis/Most_representative_words/Emotional_touch/

Generating 1 plot(s)...
  CSV results saved to: Analysis/Most_representative_words/Emotional_touch/hellinger_distance_between_betabinomial_distributions_video_23.csv
  Plot saved to: Analysis/Most_representative_words/Emotional_touch/hellinger_distance_between_betabinomial_distributions_video_23.pdf

Analysis complete!
Looks like you are using a tranform that doesn't support FancyArrowPatch, using ax.annotate instead. The arrows might strike through texts. Increasing shrinkA in arrowprops might help.
============================================================
Geometric Association Test
============================================================
Input file: Processed Data/touch_data_fixed.psv.txt
Target video(s): [24]
Text columns: ['Emotional_touch']
Minimum frequency threshold: 2
Temperature gamma: 20.0
Show N words: 20
Plots to generate: ['betabinom_hellinger']
Figure dimensions: 10.0 x 8.5 inches
Plot directory: Analysis/Most_representative_words/Emotional_touch/
Results directory for CSV files: Analysis/Most_representative_words/Emotional_touch/
Transformation dictionary: Analysis/All_words_by_frequency/word_grouping_dict_edited.json
============================================================
Loading transformation dictionary from Analysis/All_words_by_frequency/word_grouping_dict_edited.json...
  Loaded transformation dictionary with 5378 mappings
Loading data from Processed Data/touch_data_fixed.psv.txt...
Loaded 2837 rows
Setting up advanced text processing...
Loading spaCy model for enhanced negation detection...
✓ Loaded spaCy model: en_core_web_md
Concatenating text columns...
Using text columns: ['Emotional_touch']
Preprocessing text for each participant (this may take a moment)...
Grouping word variations...
Using provided transformation dictionary...
Applying groups and converting to presence/absence per participant...
Calculating geometric and probabilistic metrics...

After filtering (freq >= 2): 338 words
Created results directory: Analysis/Most_representative_words/Emotional_touch/

Generating 1 plot(s)...
  CSV results saved to: Analysis/Most_representative_words/Emotional_touch/hellinger_distance_between_betabinomial_distributions_video_24.csv
8 [ 0.79894523 -0.14376254]
10 [0.58620655 0.17326472]
  Plot saved to: Analysis/Most_representative_words/Emotional_touch/hellinger_distance_between_betabinomial_distributions_video_24.pdf

Analysis complete!
Looks like you are using a tranform that doesn't support FancyArrowPatch, using ax.annotate instead. The arrows might strike through texts. Increasing shrinkA in arrowprops might help.
============================================================
Geometric Association Test
============================================================
Input file: Processed Data/touch_data_fixed.psv.txt
Target video(s): [3, 11, 7, 19, 17, 23, 9, 1, 21, 5, 13, 15]
Text columns: ['Emotional_touch']
Minimum frequency threshold: 2
Temperature gamma: 20.0
Show N words: 20
Plots to generate: ['betabinom_hellinger']
Figure dimensions: 10.0 x 8.5 inches
Plot directory: Analysis/Most_representative_words/Emotional_touch/
Results directory for CSV files: Analysis/Most_representative_words/Emotional_touch/
Transformation dictionary: Analysis/All_words_by_frequency/word_grouping_dict_edited.json
============================================================
Loading transformation dictionary from Analysis/All_words_by_frequency/word_grouping_dict_edited.json...
  Loaded transformation dictionary with 5378 mappings
Loading data from Processed Data/touch_data_fixed.psv.txt...
Loaded 2837 rows
Setting up advanced text processing...
Loading spaCy model for enhanced negation detection...
✓ Loaded spaCy model: en_core_web_md
Concatenating text columns...
Using text columns: ['Emotional_touch']
Preprocessing text for each participant (this may take a moment)...
Grouping word variations...
Using provided transformation dictionary...
Applying groups and converting to presence/absence per participant...
Calculating geometric and probabilistic metrics...

After filtering (freq >= 2): 338 words
Created results directory: Analysis/Most_representative_words/Emotional_touch/

Generating 1 plot(s)...
  CSV results saved to: Analysis/Most_representative_words/Emotional_touch/hellinger_distance_between_betabinomial_distributions_video_3_11_7_19_17_23_9_1_21_5_13_15.csv
  Plot saved to: Analysis/Most_representative_words/Emotional_touch/hellinger_distance_between_betabinomial_distributions_video_3_11_7_19_17_23_9_1_21_5_13_15.pdf

Analysis complete!
Looks like you are using a tranform that doesn't support FancyArrowPatch, using ax.annotate instead. The arrows might strike through texts. Increasing shrinkA in arrowprops might help.
============================================================
Geometric Association Test
============================================================
Input file: Processed Data/touch_data_fixed.psv.txt
Target video(s): [16, 22, 14, 12, 4, 2, 10, 8, 24, 6, 18, 20]
Text columns: ['Emotional_touch']
Minimum frequency threshold: 2
Temperature gamma: 20.0
Show N words: 20
Plots to generate: ['betabinom_hellinger']
Figure dimensions: 10.0 x 8.5 inches
Plot directory: Analysis/Most_representative_words/Emotional_touch/
Results directory for CSV files: Analysis/Most_representative_words/Emotional_touch/
Transformation dictionary: Analysis/All_words_by_frequency/word_grouping_dict_edited.json
============================================================
Loading transformation dictionary from Analysis/All_words_by_frequency/word_grouping_dict_edited.json...
  Loaded transformation dictionary with 5378 mappings
Loading data from Processed Data/touch_data_fixed.psv.txt...
Loaded 2837 rows
Setting up advanced text processing...
Loading spaCy model for enhanced negation detection...
✓ Loaded spaCy model: en_core_web_md
Concatenating text columns...
Using text columns: ['Emotional_touch']
Preprocessing text for each participant (this may take a moment)...
Grouping word variations...
Using provided transformation dictionary...
Applying groups and converting to presence/absence per participant...
Calculating geometric and probabilistic metrics...

After filtering (freq >= 2): 338 words
Created results directory: Analysis/Most_representative_words/Emotional_touch/

Generating 1 plot(s)...
  CSV results saved to: Analysis/Most_representative_words/Emotional_touch/hellinger_distance_between_betabinomial_distributions_video_16_22_14_12_4_2_10_8_24_6_18_20.csv
  Plot saved to: Analysis/Most_representative_words/Emotional_touch/hellinger_distance_between_betabinomial_distributions_video_16_22_14_12_4_2_10_8_24_6_18_20.pdf

Analysis complete!
Looks like you are using a tranform that doesn't support FancyArrowPatch, using ax.annotate instead. The arrows might strike through texts. Increasing shrinkA in arrowprops might help.
============================================================
Geometric Association Test
============================================================
Input file: Processed Data/touch_data_fixed.psv.txt
Target video(s): [16, 14, 3, 4, 2, 17, 6, 18, 1, 5, 13, 15]
Text columns: ['Emotional_touch']
Minimum frequency threshold: 2
Temperature gamma: 20.0
Show N words: 20
Plots to generate: ['betabinom_hellinger']
Figure dimensions: 10.0 x 8.5 inches
Plot directory: Analysis/Most_representative_words/Emotional_touch/
Results directory for CSV files: Analysis/Most_representative_words/Emotional_touch/
Transformation dictionary: Analysis/All_words_by_frequency/word_grouping_dict_edited.json
============================================================
Loading transformation dictionary from Analysis/All_words_by_frequency/word_grouping_dict_edited.json...
  Loaded transformation dictionary with 5378 mappings
Loading data from Processed Data/touch_data_fixed.psv.txt...
Loaded 2837 rows
Setting up advanced text processing...
Loading spaCy model for enhanced negation detection...
✓ Loaded spaCy model: en_core_web_md
Concatenating text columns...
Using text columns: ['Emotional_touch']
Preprocessing text for each participant (this may take a moment)...
Grouping word variations...
Using provided transformation dictionary...
Applying groups and converting to presence/absence per participant...
Calculating geometric and probabilistic metrics...

After filtering (freq >= 2): 338 words
Created results directory: Analysis/Most_representative_words/Emotional_touch/

Generating 1 plot(s)...
  CSV results saved to: Analysis/Most_representative_words/Emotional_touch/hellinger_distance_between_betabinomial_distributions_video_16_14_3_4_2_17_6_18_1_5_13_15.csv
  Plot saved to: Analysis/Most_representative_words/Emotional_touch/hellinger_distance_between_betabinomial_distributions_video_16_14_3_4_2_17_6_18_1_5_13_15.pdf

Analysis complete!
Looks like you are using a tranform that doesn't support FancyArrowPatch, using ax.annotate instead. The arrows might strike through texts. Increasing shrinkA in arrowprops might help.
============================================================
Geometric Association Test
============================================================
Input file: Processed Data/touch_data_fixed.psv.txt
Target video(s): [22, 11, 12, 7, 19, 10, 8, 24, 23, 9, 20, 21]
Text columns: ['Emotional_touch']
Minimum frequency threshold: 2
Temperature gamma: 20.0
Show N words: 20
Plots to generate: ['betabinom_hellinger']
Figure dimensions: 10.0 x 8.5 inches
Plot directory: Analysis/Most_representative_words/Emotional_touch/
Results directory for CSV files: Analysis/Most_representative_words/Emotional_touch/
Transformation dictionary: Analysis/All_words_by_frequency/word_grouping_dict_edited.json
============================================================
Loading transformation dictionary from Analysis/All_words_by_frequency/word_grouping_dict_edited.json...
  Loaded transformation dictionary with 5378 mappings
Loading data from Processed Data/touch_data_fixed.psv.txt...
Loaded 2837 rows
Setting up advanced text processing...
Loading spaCy model for enhanced negation detection...
✓ Loaded spaCy model: en_core_web_md
Concatenating text columns...
Using text columns: ['Emotional_touch']
Preprocessing text for each participant (this may take a moment)...
Grouping word variations...
Using provided transformation dictionary...
Applying groups and converting to presence/absence per participant...
Calculating geometric and probabilistic metrics...

After filtering (freq >= 2): 338 words
Created results directory: Analysis/Most_representative_words/Emotional_touch/

Generating 1 plot(s)...
  CSV results saved to: Analysis/Most_representative_words/Emotional_touch/hellinger_distance_between_betabinomial_distributions_video_22_11_12_7_19_10_8_24_23_9_20_21.csv
  Plot saved to: Analysis/Most_representative_words/Emotional_touch/hellinger_distance_between_betabinomial_distributions_video_22_11_12_7_19_10_8_24_23_9_20_21.pdf

Analysis complete!
Looks like you are using a tranform that doesn't support FancyArrowPatch, using ax.annotate instead. The arrows might strike through texts. Increasing shrinkA in arrowprops might help.
============================================================
Geometric Association Test
============================================================
Input file: Processed Data/touch_data_fixed.psv.txt
Target video(s): [16, 22, 14, 19, 17, 24, 18, 23, 20, 21, 13, 15]
Text columns: ['Emotional_touch']
Minimum frequency threshold: 2
Temperature gamma: 20.0
Show N words: 20
Plots to generate: ['betabinom_hellinger']
Figure dimensions: 10.0 x 8.5 inches
Plot directory: Analysis/Most_representative_words/Emotional_touch/
Results directory for CSV files: Analysis/Most_representative_words/Emotional_touch/
Transformation dictionary: Analysis/All_words_by_frequency/word_grouping_dict_edited.json
============================================================
Loading transformation dictionary from Analysis/All_words_by_frequency/word_grouping_dict_edited.json...
  Loaded transformation dictionary with 5378 mappings
Loading data from Processed Data/touch_data_fixed.psv.txt...
Loaded 2837 rows
Setting up advanced text processing...
Loading spaCy model for enhanced negation detection...
✓ Loaded spaCy model: en_core_web_md
Concatenating text columns...
Using text columns: ['Emotional_touch']
Preprocessing text for each participant (this may take a moment)...
Grouping word variations...
Using provided transformation dictionary...
Applying groups and converting to presence/absence per participant...
Calculating geometric and probabilistic metrics...

After filtering (freq >= 2): 338 words
Created results directory: Analysis/Most_representative_words/Emotional_touch/

Generating 1 plot(s)...
  CSV results saved to: Analysis/Most_representative_words/Emotional_touch/hellinger_distance_between_betabinomial_distributions_video_16_22_14_19_17_24_18_23_20_21_13_15.csv
38 [-0.91397461  0.6886986 ]
39 [-0.11332316 -0.20875999]
  Plot saved to: Analysis/Most_representative_words/Emotional_touch/hellinger_distance_between_betabinomial_distributions_video_16_22_14_19_17_24_18_23_20_21_13_15.pdf

Analysis complete!
Looks like you are using a tranform that doesn't support FancyArrowPatch, using ax.annotate instead. The arrows might strike through texts. Increasing shrinkA in arrowprops might help.
============================================================
Geometric Association Test
============================================================
Input file: Processed Data/touch_data_fixed.psv.txt
Target video(s): [3, 11, 12, 7, 4, 2, 10, 8, 6, 9, 1, 5]
Text columns: ['Emotional_touch']
Minimum frequency threshold: 2
Temperature gamma: 20.0
Show N words: 20
Plots to generate: ['betabinom_hellinger']
Figure dimensions: 10.0 x 8.5 inches
Plot directory: Analysis/Most_representative_words/Emotional_touch/
Results directory for CSV files: Analysis/Most_representative_words/Emotional_touch/
Transformation dictionary: Analysis/All_words_by_frequency/word_grouping_dict_edited.json
============================================================
Loading transformation dictionary from Analysis/All_words_by_frequency/word_grouping_dict_edited.json...
  Loaded transformation dictionary with 5378 mappings
Loading data from Processed Data/touch_data_fixed.psv.txt...
Loaded 2837 rows
Setting up advanced text processing...
Loading spaCy model for enhanced negation detection...
✓ Loaded spaCy model: en_core_web_md
Concatenating text columns...
Using text columns: ['Emotional_touch']
Preprocessing text for each participant (this may take a moment)...
Grouping word variations...
Using provided transformation dictionary...
Applying groups and converting to presence/absence per participant...
Calculating geometric and probabilistic metrics...

After filtering (freq >= 2): 338 words
Created results directory: Analysis/Most_representative_words/Emotional_touch/

Generating 1 plot(s)...
  CSV results saved to: Analysis/Most_representative_words/Emotional_touch/hellinger_distance_between_betabinomial_distributions_video_3_11_12_7_4_2_10_8_6_9_1_5.csv
15 [0.35561365 0.54272052]
16 [0.24112144 0.16879189]
  Plot saved to: Analysis/Most_representative_words/Emotional_touch/hellinger_distance_between_betabinomial_distributions_video_3_11_12_7_4_2_10_8_6_9_1_5.pdf

Analysis complete!
Looks like you are using a tranform that doesn't support FancyArrowPatch, using ax.annotate instead. The arrows might strike through texts. Increasing shrinkA in arrowprops might help.
============================================================
Geometric Association Test
============================================================
Input file: Processed Data/touch_data_fixed.psv.txt
Target video(s): [14, 7, 19, 2, 8, 20, 1, 13]
Text columns: ['Emotional_touch']
Minimum frequency threshold: 2
Temperature gamma: 20.0
Show N words: 20
Plots to generate: ['betabinom_hellinger']
Figure dimensions: 10.0 x 8.5 inches
Plot directory: Analysis/Most_representative_words/Emotional_touch/
Results directory for CSV files: Analysis/Most_representative_words/Emotional_touch/
Transformation dictionary: Analysis/All_words_by_frequency/word_grouping_dict_edited.json
============================================================
Loading transformation dictionary from Analysis/All_words_by_frequency/word_grouping_dict_edited.json...
  Loaded transformation dictionary with 5378 mappings
Loading data from Processed Data/touch_data_fixed.psv.txt...
Loaded 2837 rows
Setting up advanced text processing...
Loading spaCy model for enhanced negation detection...
✓ Loaded spaCy model: en_core_web_md
Concatenating text columns...
Using text columns: ['Emotional_touch']
Preprocessing text for each participant (this may take a moment)...
Grouping word variations...
Using provided transformation dictionary...
Applying groups and converting to presence/absence per participant...
Calculating geometric and probabilistic metrics...

After filtering (freq >= 2): 338 words
Created results directory: Analysis/Most_representative_words/Emotional_touch/

Generating 1 plot(s)...
  CSV results saved to: Analysis/Most_representative_words/Emotional_touch/hellinger_distance_between_betabinomial_distributions_video_14_7_19_2_8_20_1_13.csv
  Plot saved to: Analysis/Most_representative_words/Emotional_touch/hellinger_distance_between_betabinomial_distributions_video_14_7_19_2_8_20_1_13.pdf

Analysis complete!
Looks like you are using a tranform that doesn't support FancyArrowPatch, using ax.annotate instead. The arrows might strike through texts. Increasing shrinkA in arrowprops might help.
============================================================
Geometric Association Test
============================================================
Input file: Processed Data/touch_data_fixed.psv.txt
Target video(s): [16, 22, 3, 4, 10, 9, 21, 15]
Text columns: ['Emotional_touch']
Minimum frequency threshold: 2
Temperature gamma: 20.0
Show N words: 20
Plots to generate: ['betabinom_hellinger']
Figure dimensions: 10.0 x 8.5 inches
Plot directory: Analysis/Most_representative_words/Emotional_touch/
Results directory for CSV files: Analysis/Most_representative_words/Emotional_touch/
Transformation dictionary: Analysis/All_words_by_frequency/word_grouping_dict_edited.json
============================================================
Loading transformation dictionary from Analysis/All_words_by_frequency/word_grouping_dict_edited.json...
  Loaded transformation dictionary with 5378 mappings
Loading data from Processed Data/touch_data_fixed.psv.txt...
Loaded 2837 rows
Setting up advanced text processing...
Loading spaCy model for enhanced negation detection...
✓ Loaded spaCy model: en_core_web_md
Concatenating text columns...
Using text columns: ['Emotional_touch']
Preprocessing text for each participant (this may take a moment)...
Grouping word variations...
Using provided transformation dictionary...
Applying groups and converting to presence/absence per participant...
Calculating geometric and probabilistic metrics...

After filtering (freq >= 2): 338 words
Created results directory: Analysis/Most_representative_words/Emotional_touch/

Generating 1 plot(s)...
  CSV results saved to: Analysis/Most_representative_words/Emotional_touch/hellinger_distance_between_betabinomial_distributions_video_16_22_3_4_10_9_21_15.csv
9 [-0.46110012  0.57999097]
14 [0.05243051 0.46870915]
  Plot saved to: Analysis/Most_representative_words/Emotional_touch/hellinger_distance_between_betabinomial_distributions_video_16_22_3_4_10_9_21_15.pdf

Analysis complete!
Looks like you are using a tranform that doesn't support FancyArrowPatch, using ax.annotate instead. The arrows might strike through texts. Increasing shrinkA in arrowprops might help.
============================================================
Geometric Association Test
============================================================
Input file: Processed Data/touch_data_fixed.psv.txt
Target video(s): [11, 12, 17, 24, 6, 18, 23, 5]
Text columns: ['Emotional_touch']
Minimum frequency threshold: 2
Temperature gamma: 20.0
Show N words: 20
Plots to generate: ['betabinom_hellinger']
Figure dimensions: 10.0 x 8.5 inches
Plot directory: Analysis/Most_representative_words/Emotional_touch/
Results directory for CSV files: Analysis/Most_representative_words/Emotional_touch/
Transformation dictionary: Analysis/All_words_by_frequency/word_grouping_dict_edited.json
============================================================
Loading transformation dictionary from Analysis/All_words_by_frequency/word_grouping_dict_edited.json...
  Loaded transformation dictionary with 5378 mappings
Loading data from Processed Data/touch_data_fixed.psv.txt...
Loaded 2837 rows
Setting up advanced text processing...
Loading spaCy model for enhanced negation detection...
✓ Loaded spaCy model: en_core_web_md
Concatenating text columns...
Using text columns: ['Emotional_touch']
Preprocessing text for each participant (this may take a moment)...
Grouping word variations...
Using provided transformation dictionary...
Applying groups and converting to presence/absence per participant...
Calculating geometric and probabilistic metrics...

After filtering (freq >= 2): 338 words
Created results directory: Analysis/Most_representative_words/Emotional_touch/

Generating 1 plot(s)...
  CSV results saved to: Analysis/Most_representative_words/Emotional_touch/hellinger_distance_between_betabinomial_distributions_video_11_12_17_24_6_18_23_5.csv
  Plot saved to: Analysis/Most_representative_words/Emotional_touch/hellinger_distance_between_betabinomial_distributions_video_11_12_17_24_6_18_23_5.pdf

Analysis complete!

```


Data aggregation:

```text
Using ['Word', 'Total_Freq'] as ID columns
Using Betabinom_Hellinger as variable columns

Compiled DataFrame:
           Word  ...  videos_11_12_17_24_6_18_23_5
0     attention  ...                      0.666095
1      grabbing  ...                      0.677977
2       focused  ...                      0.265069
3         wants  ...                      0.265069
4      hesitant  ...                     -0.411799
..          ...  ...                           ...
333  meaningful  ...                     -0.389013
334     calming  ...                     -0.803592
335     sensual  ...                     -0.925443
336  comforting  ...                     -0.958930
337    friendly  ...                     -0.086970

[338 rows x 36 columns]

Saving to CSV: Analysis/Most_representative_words/Emotional_touch/aggregated_data.csv.txt

```
