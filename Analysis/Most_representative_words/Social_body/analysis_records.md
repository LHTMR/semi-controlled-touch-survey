
# Analysis Record

### Aim

- Determine what are the most (and least) representative words for each video
- Compare different groups of videos per touch parameters (light vs. strong; finger vs. hand; 3cm/s vs. 9cm/s vs. 18 cm/s) and see if the calculated statistic can separate words in a way that appears to reflect the difference in modalities.
- Specific focus on `Social_body` to determine whether some touches are most associated with specific social relations

### Command line

Executed from the root of the directory:

```bash
for j in $(seq 24) "3 11 7 19 17 23 9 1 21 5 13 15" "16 22 14 12 4 2 10 8 24 6 18 20" "16 14 3 4 2 17 6 18 1 5 13 15" "22 11 12 7 19 10 8 24 23 9 20 21" "16 22 14 19 17 24 18 23 20 21 13 15" "3 11 12 7 4 2 10 8 6 9 1 5" "14 7 19 2 8 20 1 13" "16 22 3 4 10 9 21 15" "11 12 17 24 6 18 23 5"; do python3 Analysis/Scripts/geometric_association_test.py --input "Processed Data/touch_data_fixed.psv.txt" --transformation-dict Analysis/All_words_by_frequency/word_grouping_dict_edited.json --plot-dir Analysis/Most_representative_words/Social_body/ --results-dir Analysis/Most_representative_words/Social_body/ --min-freq 2 --show-words 20 --text-columns Social_body --plots "betabinom_hellinger" --target-video $j ; done
```


Data aggregation:

```bash
python3 Analysis/Scripts/aggregate_csv_data.py --input-dir Analysis/Most_representative_words/Social_body
```


Data cleanup:

```bash
rm Analysis/Most_representative_words/Social_body/hellinger_distance_between_betabinomial_distributions_video_*.csv
```


### Execution logs

```text
Looks like you are using a tranform that doesn't support FancyArrowPatch, using ax.annotate instead. The arrows might strike through texts. Increasing shrinkA in arrowprops might help.
============================================================
Geometric Association Test
============================================================
Input file: Processed Data/touch_data_fixed.psv.txt
Target video(s): [1]
Text columns: ['Social_body']
Minimum frequency threshold: 2
Temperature gamma: 20.0
Show N words: 20
Plots to generate: ['betabinom_hellinger']
Figure dimensions: 10.0 x 8.5 inches
Plot directory: Analysis/Most_representative_words/Social_body/
Results directory for CSV files: Analysis/Most_representative_words/Social_body/
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
Using text columns: ['Social_body']
Preprocessing text for each participant (this may take a moment)...
    ⚠️  Text filtered to empty: 'my  back...' -> spaCy tokens: [' ']
    ⚠️  Text filtered to empty: 's...' -> spaCy tokens: ['s']
Grouping word variations...
Using provided transformation dictionary...
Applying groups and converting to presence/absence per participant...
Calculating geometric and probabilistic metrics...

After filtering (freq >= 2): 94 words
Created results directory: Analysis/Most_representative_words/Social_body/

Generating 1 plot(s)...
  CSV results saved to: Analysis/Most_representative_words/Social_body/hellinger_distance_between_betabinomial_distributions_video_1.csv
  Plot saved to: Analysis/Most_representative_words/Social_body/hellinger_distance_between_betabinomial_distributions_video_1.pdf

Analysis complete!
Looks like you are using a tranform that doesn't support FancyArrowPatch, using ax.annotate instead. The arrows might strike through texts. Increasing shrinkA in arrowprops might help.
============================================================
Geometric Association Test
============================================================
Input file: Processed Data/touch_data_fixed.psv.txt
Target video(s): [2]
Text columns: ['Social_body']
Minimum frequency threshold: 2
Temperature gamma: 20.0
Show N words: 20
Plots to generate: ['betabinom_hellinger']
Figure dimensions: 10.0 x 8.5 inches
Plot directory: Analysis/Most_representative_words/Social_body/
Results directory for CSV files: Analysis/Most_representative_words/Social_body/
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
Using text columns: ['Social_body']
Preprocessing text for each participant (this may take a moment)...
    ⚠️  Text filtered to empty: 'my  back...' -> spaCy tokens: [' ']
    ⚠️  Text filtered to empty: 's...' -> spaCy tokens: ['s']
Grouping word variations...
Using provided transformation dictionary...
Applying groups and converting to presence/absence per participant...
Calculating geometric and probabilistic metrics...

After filtering (freq >= 2): 94 words
Created results directory: Analysis/Most_representative_words/Social_body/

Generating 1 plot(s)...
  CSV results saved to: Analysis/Most_representative_words/Social_body/hellinger_distance_between_betabinomial_distributions_video_2.csv
  Plot saved to: Analysis/Most_representative_words/Social_body/hellinger_distance_between_betabinomial_distributions_video_2.pdf

Analysis complete!
Looks like you are using a tranform that doesn't support FancyArrowPatch, using ax.annotate instead. The arrows might strike through texts. Increasing shrinkA in arrowprops might help.
============================================================
Geometric Association Test
============================================================
Input file: Processed Data/touch_data_fixed.psv.txt
Target video(s): [3]
Text columns: ['Social_body']
Minimum frequency threshold: 2
Temperature gamma: 20.0
Show N words: 20
Plots to generate: ['betabinom_hellinger']
Figure dimensions: 10.0 x 8.5 inches
Plot directory: Analysis/Most_representative_words/Social_body/
Results directory for CSV files: Analysis/Most_representative_words/Social_body/
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
Using text columns: ['Social_body']
Preprocessing text for each participant (this may take a moment)...
    ⚠️  Text filtered to empty: 'my  back...' -> spaCy tokens: [' ']
    ⚠️  Text filtered to empty: 's...' -> spaCy tokens: ['s']
Grouping word variations...
Using provided transformation dictionary...
Applying groups and converting to presence/absence per participant...
Calculating geometric and probabilistic metrics...

After filtering (freq >= 2): 94 words
Created results directory: Analysis/Most_representative_words/Social_body/

Generating 1 plot(s)...
  CSV results saved to: Analysis/Most_representative_words/Social_body/hellinger_distance_between_betabinomial_distributions_video_3.csv
  Plot saved to: Analysis/Most_representative_words/Social_body/hellinger_distance_between_betabinomial_distributions_video_3.pdf

Analysis complete!
Looks like you are using a tranform that doesn't support FancyArrowPatch, using ax.annotate instead. The arrows might strike through texts. Increasing shrinkA in arrowprops might help.
============================================================
Geometric Association Test
============================================================
Input file: Processed Data/touch_data_fixed.psv.txt
Target video(s): [4]
Text columns: ['Social_body']
Minimum frequency threshold: 2
Temperature gamma: 20.0
Show N words: 20
Plots to generate: ['betabinom_hellinger']
Figure dimensions: 10.0 x 8.5 inches
Plot directory: Analysis/Most_representative_words/Social_body/
Results directory for CSV files: Analysis/Most_representative_words/Social_body/
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
Using text columns: ['Social_body']
Preprocessing text for each participant (this may take a moment)...
    ⚠️  Text filtered to empty: 'my  back...' -> spaCy tokens: [' ']
    ⚠️  Text filtered to empty: 's...' -> spaCy tokens: ['s']
Grouping word variations...
Using provided transformation dictionary...
Applying groups and converting to presence/absence per participant...
Calculating geometric and probabilistic metrics...

After filtering (freq >= 2): 94 words
Created results directory: Analysis/Most_representative_words/Social_body/

Generating 1 plot(s)...
  CSV results saved to: Analysis/Most_representative_words/Social_body/hellinger_distance_between_betabinomial_distributions_video_4.csv
  Plot saved to: Analysis/Most_representative_words/Social_body/hellinger_distance_between_betabinomial_distributions_video_4.pdf

Analysis complete!
Looks like you are using a tranform that doesn't support FancyArrowPatch, using ax.annotate instead. The arrows might strike through texts. Increasing shrinkA in arrowprops might help.
============================================================
Geometric Association Test
============================================================
Input file: Processed Data/touch_data_fixed.psv.txt
Target video(s): [5]
Text columns: ['Social_body']
Minimum frequency threshold: 2
Temperature gamma: 20.0
Show N words: 20
Plots to generate: ['betabinom_hellinger']
Figure dimensions: 10.0 x 8.5 inches
Plot directory: Analysis/Most_representative_words/Social_body/
Results directory for CSV files: Analysis/Most_representative_words/Social_body/
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
Using text columns: ['Social_body']
Preprocessing text for each participant (this may take a moment)...
    ⚠️  Text filtered to empty: 'my  back...' -> spaCy tokens: [' ']
    ⚠️  Text filtered to empty: 's...' -> spaCy tokens: ['s']
Grouping word variations...
Using provided transformation dictionary...
Applying groups and converting to presence/absence per participant...
Calculating geometric and probabilistic metrics...

After filtering (freq >= 2): 94 words
Created results directory: Analysis/Most_representative_words/Social_body/

Generating 1 plot(s)...
  CSV results saved to: Analysis/Most_representative_words/Social_body/hellinger_distance_between_betabinomial_distributions_video_5.csv
  Plot saved to: Analysis/Most_representative_words/Social_body/hellinger_distance_between_betabinomial_distributions_video_5.pdf

Analysis complete!
Looks like you are using a tranform that doesn't support FancyArrowPatch, using ax.annotate instead. The arrows might strike through texts. Increasing shrinkA in arrowprops might help.
============================================================
Geometric Association Test
============================================================
Input file: Processed Data/touch_data_fixed.psv.txt
Target video(s): [6]
Text columns: ['Social_body']
Minimum frequency threshold: 2
Temperature gamma: 20.0
Show N words: 20
Plots to generate: ['betabinom_hellinger']
Figure dimensions: 10.0 x 8.5 inches
Plot directory: Analysis/Most_representative_words/Social_body/
Results directory for CSV files: Analysis/Most_representative_words/Social_body/
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
Using text columns: ['Social_body']
Preprocessing text for each participant (this may take a moment)...
    ⚠️  Text filtered to empty: 'my  back...' -> spaCy tokens: [' ']
    ⚠️  Text filtered to empty: 's...' -> spaCy tokens: ['s']
Grouping word variations...
Using provided transformation dictionary...
Applying groups and converting to presence/absence per participant...
Calculating geometric and probabilistic metrics...

After filtering (freq >= 2): 94 words
Created results directory: Analysis/Most_representative_words/Social_body/

Generating 1 plot(s)...
  CSV results saved to: Analysis/Most_representative_words/Social_body/hellinger_distance_between_betabinomial_distributions_video_6.csv
3 [ 0.84729896 -0.25163933]
4 [-0.86922998 -0.06213966]
  Plot saved to: Analysis/Most_representative_words/Social_body/hellinger_distance_between_betabinomial_distributions_video_6.pdf

Analysis complete!
Looks like you are using a tranform that doesn't support FancyArrowPatch, using ax.annotate instead. The arrows might strike through texts. Increasing shrinkA in arrowprops might help.
============================================================
Geometric Association Test
============================================================
Input file: Processed Data/touch_data_fixed.psv.txt
Target video(s): [7]
Text columns: ['Social_body']
Minimum frequency threshold: 2
Temperature gamma: 20.0
Show N words: 20
Plots to generate: ['betabinom_hellinger']
Figure dimensions: 10.0 x 8.5 inches
Plot directory: Analysis/Most_representative_words/Social_body/
Results directory for CSV files: Analysis/Most_representative_words/Social_body/
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
Using text columns: ['Social_body']
Preprocessing text for each participant (this may take a moment)...
    ⚠️  Text filtered to empty: 'my  back...' -> spaCy tokens: [' ']
    ⚠️  Text filtered to empty: 's...' -> spaCy tokens: ['s']
Grouping word variations...
Using provided transformation dictionary...
Applying groups and converting to presence/absence per participant...
Calculating geometric and probabilistic metrics...

After filtering (freq >= 2): 94 words
Created results directory: Analysis/Most_representative_words/Social_body/

Generating 1 plot(s)...
  CSV results saved to: Analysis/Most_representative_words/Social_body/hellinger_distance_between_betabinomial_distributions_video_7.csv
  Plot saved to: Analysis/Most_representative_words/Social_body/hellinger_distance_between_betabinomial_distributions_video_7.pdf

Analysis complete!
Looks like you are using a tranform that doesn't support FancyArrowPatch, using ax.annotate instead. The arrows might strike through texts. Increasing shrinkA in arrowprops might help.
============================================================
Geometric Association Test
============================================================
Input file: Processed Data/touch_data_fixed.psv.txt
Target video(s): [8]
Text columns: ['Social_body']
Minimum frequency threshold: 2
Temperature gamma: 20.0
Show N words: 20
Plots to generate: ['betabinom_hellinger']
Figure dimensions: 10.0 x 8.5 inches
Plot directory: Analysis/Most_representative_words/Social_body/
Results directory for CSV files: Analysis/Most_representative_words/Social_body/
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
Using text columns: ['Social_body']
Preprocessing text for each participant (this may take a moment)...
    ⚠️  Text filtered to empty: 'my  back...' -> spaCy tokens: [' ']
    ⚠️  Text filtered to empty: 's...' -> spaCy tokens: ['s']
Grouping word variations...
Using provided transformation dictionary...
Applying groups and converting to presence/absence per participant...
Calculating geometric and probabilistic metrics...

After filtering (freq >= 2): 94 words
Created results directory: Analysis/Most_representative_words/Social_body/

Generating 1 plot(s)...
  CSV results saved to: Analysis/Most_representative_words/Social_body/hellinger_distance_between_betabinomial_distributions_video_8.csv
  Plot saved to: Analysis/Most_representative_words/Social_body/hellinger_distance_between_betabinomial_distributions_video_8.pdf

Analysis complete!
Looks like you are using a tranform that doesn't support FancyArrowPatch, using ax.annotate instead. The arrows might strike through texts. Increasing shrinkA in arrowprops might help.
============================================================
Geometric Association Test
============================================================
Input file: Processed Data/touch_data_fixed.psv.txt
Target video(s): [9]
Text columns: ['Social_body']
Minimum frequency threshold: 2
Temperature gamma: 20.0
Show N words: 20
Plots to generate: ['betabinom_hellinger']
Figure dimensions: 10.0 x 8.5 inches
Plot directory: Analysis/Most_representative_words/Social_body/
Results directory for CSV files: Analysis/Most_representative_words/Social_body/
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
Using text columns: ['Social_body']
Preprocessing text for each participant (this may take a moment)...
    ⚠️  Text filtered to empty: 'my  back...' -> spaCy tokens: [' ']
    ⚠️  Text filtered to empty: 's...' -> spaCy tokens: ['s']
Grouping word variations...
Using provided transformation dictionary...
Applying groups and converting to presence/absence per participant...
Calculating geometric and probabilistic metrics...

After filtering (freq >= 2): 94 words
Created results directory: Analysis/Most_representative_words/Social_body/

Generating 1 plot(s)...
  CSV results saved to: Analysis/Most_representative_words/Social_body/hellinger_distance_between_betabinomial_distributions_video_9.csv
  Plot saved to: Analysis/Most_representative_words/Social_body/hellinger_distance_between_betabinomial_distributions_video_9.pdf

Analysis complete!
Looks like you are using a tranform that doesn't support FancyArrowPatch, using ax.annotate instead. The arrows might strike through texts. Increasing shrinkA in arrowprops might help.
============================================================
Geometric Association Test
============================================================
Input file: Processed Data/touch_data_fixed.psv.txt
Target video(s): [10]
Text columns: ['Social_body']
Minimum frequency threshold: 2
Temperature gamma: 20.0
Show N words: 20
Plots to generate: ['betabinom_hellinger']
Figure dimensions: 10.0 x 8.5 inches
Plot directory: Analysis/Most_representative_words/Social_body/
Results directory for CSV files: Analysis/Most_representative_words/Social_body/
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
Using text columns: ['Social_body']
Preprocessing text for each participant (this may take a moment)...
    ⚠️  Text filtered to empty: 'my  back...' -> spaCy tokens: [' ']
    ⚠️  Text filtered to empty: 's...' -> spaCy tokens: ['s']
Grouping word variations...
Using provided transformation dictionary...
Applying groups and converting to presence/absence per participant...
Calculating geometric and probabilistic metrics...

After filtering (freq >= 2): 94 words
Created results directory: Analysis/Most_representative_words/Social_body/

Generating 1 plot(s)...
  CSV results saved to: Analysis/Most_representative_words/Social_body/hellinger_distance_between_betabinomial_distributions_video_10.csv
12 [-0.96286798  0.03987998]
19 [0.66065489 0.68134765]
12 [-0.76455626  0.09641871]
18 [0.20013913 0.29276634]
  Plot saved to: Analysis/Most_representative_words/Social_body/hellinger_distance_between_betabinomial_distributions_video_10.pdf

Analysis complete!
Looks like you are using a tranform that doesn't support FancyArrowPatch, using ax.annotate instead. The arrows might strike through texts. Increasing shrinkA in arrowprops might help.
============================================================
Geometric Association Test
============================================================
Input file: Processed Data/touch_data_fixed.psv.txt
Target video(s): [11]
Text columns: ['Social_body']
Minimum frequency threshold: 2
Temperature gamma: 20.0
Show N words: 20
Plots to generate: ['betabinom_hellinger']
Figure dimensions: 10.0 x 8.5 inches
Plot directory: Analysis/Most_representative_words/Social_body/
Results directory for CSV files: Analysis/Most_representative_words/Social_body/
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
Using text columns: ['Social_body']
Preprocessing text for each participant (this may take a moment)...
    ⚠️  Text filtered to empty: 'my  back...' -> spaCy tokens: [' ']
    ⚠️  Text filtered to empty: 's...' -> spaCy tokens: ['s']
Grouping word variations...
Using provided transformation dictionary...
Applying groups and converting to presence/absence per participant...
Calculating geometric and probabilistic metrics...

After filtering (freq >= 2): 94 words
Created results directory: Analysis/Most_representative_words/Social_body/

Generating 1 plot(s)...
  CSV results saved to: Analysis/Most_representative_words/Social_body/hellinger_distance_between_betabinomial_distributions_video_11.csv
12 [ 0.06475296 -0.45228579]
15 [0.63531832 0.73388493]
  Plot saved to: Analysis/Most_representative_words/Social_body/hellinger_distance_between_betabinomial_distributions_video_11.pdf

Analysis complete!
Looks like you are using a tranform that doesn't support FancyArrowPatch, using ax.annotate instead. The arrows might strike through texts. Increasing shrinkA in arrowprops might help.
============================================================
Geometric Association Test
============================================================
Input file: Processed Data/touch_data_fixed.psv.txt
Target video(s): [12]
Text columns: ['Social_body']
Minimum frequency threshold: 2
Temperature gamma: 20.0
Show N words: 20
Plots to generate: ['betabinom_hellinger']
Figure dimensions: 10.0 x 8.5 inches
Plot directory: Analysis/Most_representative_words/Social_body/
Results directory for CSV files: Analysis/Most_representative_words/Social_body/
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
Using text columns: ['Social_body']
Preprocessing text for each participant (this may take a moment)...
    ⚠️  Text filtered to empty: 'my  back...' -> spaCy tokens: [' ']
    ⚠️  Text filtered to empty: 's...' -> spaCy tokens: ['s']
Grouping word variations...
Using provided transformation dictionary...
Applying groups and converting to presence/absence per participant...
Calculating geometric and probabilistic metrics...

After filtering (freq >= 2): 94 words
Created results directory: Analysis/Most_representative_words/Social_body/

Generating 1 plot(s)...
  CSV results saved to: Analysis/Most_representative_words/Social_body/hellinger_distance_between_betabinomial_distributions_video_12.csv
  Plot saved to: Analysis/Most_representative_words/Social_body/hellinger_distance_between_betabinomial_distributions_video_12.pdf

Analysis complete!
Looks like you are using a tranform that doesn't support FancyArrowPatch, using ax.annotate instead. The arrows might strike through texts. Increasing shrinkA in arrowprops might help.
============================================================
Geometric Association Test
============================================================
Input file: Processed Data/touch_data_fixed.psv.txt
Target video(s): [13]
Text columns: ['Social_body']
Minimum frequency threshold: 2
Temperature gamma: 20.0
Show N words: 20
Plots to generate: ['betabinom_hellinger']
Figure dimensions: 10.0 x 8.5 inches
Plot directory: Analysis/Most_representative_words/Social_body/
Results directory for CSV files: Analysis/Most_representative_words/Social_body/
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
Using text columns: ['Social_body']
Preprocessing text for each participant (this may take a moment)...
    ⚠️  Text filtered to empty: 'my  back...' -> spaCy tokens: [' ']
    ⚠️  Text filtered to empty: 's...' -> spaCy tokens: ['s']
Grouping word variations...
Using provided transformation dictionary...
Applying groups and converting to presence/absence per participant...
Calculating geometric and probabilistic metrics...

After filtering (freq >= 2): 94 words
Created results directory: Analysis/Most_representative_words/Social_body/

Generating 1 plot(s)...
  CSV results saved to: Analysis/Most_representative_words/Social_body/hellinger_distance_between_betabinomial_distributions_video_13.csv
  Plot saved to: Analysis/Most_representative_words/Social_body/hellinger_distance_between_betabinomial_distributions_video_13.pdf

Analysis complete!
Looks like you are using a tranform that doesn't support FancyArrowPatch, using ax.annotate instead. The arrows might strike through texts. Increasing shrinkA in arrowprops might help.
============================================================
Geometric Association Test
============================================================
Input file: Processed Data/touch_data_fixed.psv.txt
Target video(s): [14]
Text columns: ['Social_body']
Minimum frequency threshold: 2
Temperature gamma: 20.0
Show N words: 20
Plots to generate: ['betabinom_hellinger']
Figure dimensions: 10.0 x 8.5 inches
Plot directory: Analysis/Most_representative_words/Social_body/
Results directory for CSV files: Analysis/Most_representative_words/Social_body/
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
Using text columns: ['Social_body']
Preprocessing text for each participant (this may take a moment)...
    ⚠️  Text filtered to empty: 'my  back...' -> spaCy tokens: [' ']
    ⚠️  Text filtered to empty: 's...' -> spaCy tokens: ['s']
Grouping word variations...
Using provided transformation dictionary...
Applying groups and converting to presence/absence per participant...
Calculating geometric and probabilistic metrics...

After filtering (freq >= 2): 94 words
Created results directory: Analysis/Most_representative_words/Social_body/

Generating 1 plot(s)...
  CSV results saved to: Analysis/Most_representative_words/Social_body/hellinger_distance_between_betabinomial_distributions_video_14.csv
  Plot saved to: Analysis/Most_representative_words/Social_body/hellinger_distance_between_betabinomial_distributions_video_14.pdf

Analysis complete!
Looks like you are using a tranform that doesn't support FancyArrowPatch, using ax.annotate instead. The arrows might strike through texts. Increasing shrinkA in arrowprops might help.
============================================================
Geometric Association Test
============================================================
Input file: Processed Data/touch_data_fixed.psv.txt
Target video(s): [15]
Text columns: ['Social_body']
Minimum frequency threshold: 2
Temperature gamma: 20.0
Show N words: 20
Plots to generate: ['betabinom_hellinger']
Figure dimensions: 10.0 x 8.5 inches
Plot directory: Analysis/Most_representative_words/Social_body/
Results directory for CSV files: Analysis/Most_representative_words/Social_body/
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
Using text columns: ['Social_body']
Preprocessing text for each participant (this may take a moment)...
    ⚠️  Text filtered to empty: 'my  back...' -> spaCy tokens: [' ']
    ⚠️  Text filtered to empty: 's...' -> spaCy tokens: ['s']
Grouping word variations...
Using provided transformation dictionary...
Applying groups and converting to presence/absence per participant...
Calculating geometric and probabilistic metrics...

After filtering (freq >= 2): 94 words
Created results directory: Analysis/Most_representative_words/Social_body/

Generating 1 plot(s)...
  CSV results saved to: Analysis/Most_representative_words/Social_body/hellinger_distance_between_betabinomial_distributions_video_15.csv
13 [-0.65213419 -0.0735589 ]
17 [-0.24070004 -0.75613719]
  Plot saved to: Analysis/Most_representative_words/Social_body/hellinger_distance_between_betabinomial_distributions_video_15.pdf

Analysis complete!
Looks like you are using a tranform that doesn't support FancyArrowPatch, using ax.annotate instead. The arrows might strike through texts. Increasing shrinkA in arrowprops might help.
============================================================
Geometric Association Test
============================================================
Input file: Processed Data/touch_data_fixed.psv.txt
Target video(s): [16]
Text columns: ['Social_body']
Minimum frequency threshold: 2
Temperature gamma: 20.0
Show N words: 20
Plots to generate: ['betabinom_hellinger']
Figure dimensions: 10.0 x 8.5 inches
Plot directory: Analysis/Most_representative_words/Social_body/
Results directory for CSV files: Analysis/Most_representative_words/Social_body/
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
Using text columns: ['Social_body']
Preprocessing text for each participant (this may take a moment)...
    ⚠️  Text filtered to empty: 'my  back...' -> spaCy tokens: [' ']
    ⚠️  Text filtered to empty: 's...' -> spaCy tokens: ['s']
Grouping word variations...
Using provided transformation dictionary...
Applying groups and converting to presence/absence per participant...
Calculating geometric and probabilistic metrics...

After filtering (freq >= 2): 94 words
Created results directory: Analysis/Most_representative_words/Social_body/

Generating 1 plot(s)...
  CSV results saved to: Analysis/Most_representative_words/Social_body/hellinger_distance_between_betabinomial_distributions_video_16.csv
  Plot saved to: Analysis/Most_representative_words/Social_body/hellinger_distance_between_betabinomial_distributions_video_16.pdf

Analysis complete!
Looks like you are using a tranform that doesn't support FancyArrowPatch, using ax.annotate instead. The arrows might strike through texts. Increasing shrinkA in arrowprops might help.
============================================================
Geometric Association Test
============================================================
Input file: Processed Data/touch_data_fixed.psv.txt
Target video(s): [17]
Text columns: ['Social_body']
Minimum frequency threshold: 2
Temperature gamma: 20.0
Show N words: 20
Plots to generate: ['betabinom_hellinger']
Figure dimensions: 10.0 x 8.5 inches
Plot directory: Analysis/Most_representative_words/Social_body/
Results directory for CSV files: Analysis/Most_representative_words/Social_body/
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
Using text columns: ['Social_body']
Preprocessing text for each participant (this may take a moment)...
    ⚠️  Text filtered to empty: 'my  back...' -> spaCy tokens: [' ']
    ⚠️  Text filtered to empty: 's...' -> spaCy tokens: ['s']
Grouping word variations...
Using provided transformation dictionary...
Applying groups and converting to presence/absence per participant...
Calculating geometric and probabilistic metrics...

After filtering (freq >= 2): 94 words
Created results directory: Analysis/Most_representative_words/Social_body/

Generating 1 plot(s)...
  CSV results saved to: Analysis/Most_representative_words/Social_body/hellinger_distance_between_betabinomial_distributions_video_17.csv
5 [0.0028127  0.61499788]
6 [ 0.42547723 -0.33867716]
  Plot saved to: Analysis/Most_representative_words/Social_body/hellinger_distance_between_betabinomial_distributions_video_17.pdf

Analysis complete!
Looks like you are using a tranform that doesn't support FancyArrowPatch, using ax.annotate instead. The arrows might strike through texts. Increasing shrinkA in arrowprops might help.
============================================================
Geometric Association Test
============================================================
Input file: Processed Data/touch_data_fixed.psv.txt
Target video(s): [18]
Text columns: ['Social_body']
Minimum frequency threshold: 2
Temperature gamma: 20.0
Show N words: 20
Plots to generate: ['betabinom_hellinger']
Figure dimensions: 10.0 x 8.5 inches
Plot directory: Analysis/Most_representative_words/Social_body/
Results directory for CSV files: Analysis/Most_representative_words/Social_body/
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
Using text columns: ['Social_body']
Preprocessing text for each participant (this may take a moment)...
    ⚠️  Text filtered to empty: 'my  back...' -> spaCy tokens: [' ']
    ⚠️  Text filtered to empty: 's...' -> spaCy tokens: ['s']
Grouping word variations...
Using provided transformation dictionary...
Applying groups and converting to presence/absence per participant...
Calculating geometric and probabilistic metrics...

After filtering (freq >= 2): 94 words
Created results directory: Analysis/Most_representative_words/Social_body/

Generating 1 plot(s)...
  CSV results saved to: Analysis/Most_representative_words/Social_body/hellinger_distance_between_betabinomial_distributions_video_18.csv
  Plot saved to: Analysis/Most_representative_words/Social_body/hellinger_distance_between_betabinomial_distributions_video_18.pdf

Analysis complete!
Looks like you are using a tranform that doesn't support FancyArrowPatch, using ax.annotate instead. The arrows might strike through texts. Increasing shrinkA in arrowprops might help.
============================================================
Geometric Association Test
============================================================
Input file: Processed Data/touch_data_fixed.psv.txt
Target video(s): [19]
Text columns: ['Social_body']
Minimum frequency threshold: 2
Temperature gamma: 20.0
Show N words: 20
Plots to generate: ['betabinom_hellinger']
Figure dimensions: 10.0 x 8.5 inches
Plot directory: Analysis/Most_representative_words/Social_body/
Results directory for CSV files: Analysis/Most_representative_words/Social_body/
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
Using text columns: ['Social_body']
Preprocessing text for each participant (this may take a moment)...
    ⚠️  Text filtered to empty: 'my  back...' -> spaCy tokens: [' ']
    ⚠️  Text filtered to empty: 's...' -> spaCy tokens: ['s']
Grouping word variations...
Using provided transformation dictionary...
Applying groups and converting to presence/absence per participant...
Calculating geometric and probabilistic metrics...

After filtering (freq >= 2): 94 words
Created results directory: Analysis/Most_representative_words/Social_body/

Generating 1 plot(s)...
  CSV results saved to: Analysis/Most_representative_words/Social_body/hellinger_distance_between_betabinomial_distributions_video_19.csv
16 [-0.85187184  0.80005827]
18 [-0.07642892  0.26920557]
  Plot saved to: Analysis/Most_representative_words/Social_body/hellinger_distance_between_betabinomial_distributions_video_19.pdf

Analysis complete!
Looks like you are using a tranform that doesn't support FancyArrowPatch, using ax.annotate instead. The arrows might strike through texts. Increasing shrinkA in arrowprops might help.
============================================================
Geometric Association Test
============================================================
Input file: Processed Data/touch_data_fixed.psv.txt
Target video(s): [20]
Text columns: ['Social_body']
Minimum frequency threshold: 2
Temperature gamma: 20.0
Show N words: 20
Plots to generate: ['betabinom_hellinger']
Figure dimensions: 10.0 x 8.5 inches
Plot directory: Analysis/Most_representative_words/Social_body/
Results directory for CSV files: Analysis/Most_representative_words/Social_body/
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
Using text columns: ['Social_body']
Preprocessing text for each participant (this may take a moment)...
    ⚠️  Text filtered to empty: 'my  back...' -> spaCy tokens: [' ']
    ⚠️  Text filtered to empty: 's...' -> spaCy tokens: ['s']
Grouping word variations...
Using provided transformation dictionary...
Applying groups and converting to presence/absence per participant...
Calculating geometric and probabilistic metrics...

After filtering (freq >= 2): 94 words
Created results directory: Analysis/Most_representative_words/Social_body/

Generating 1 plot(s)...
  CSV results saved to: Analysis/Most_representative_words/Social_body/hellinger_distance_between_betabinomial_distributions_video_20.csv
  Plot saved to: Analysis/Most_representative_words/Social_body/hellinger_distance_between_betabinomial_distributions_video_20.pdf

Analysis complete!
Looks like you are using a tranform that doesn't support FancyArrowPatch, using ax.annotate instead. The arrows might strike through texts. Increasing shrinkA in arrowprops might help.
============================================================
Geometric Association Test
============================================================
Input file: Processed Data/touch_data_fixed.psv.txt
Target video(s): [21]
Text columns: ['Social_body']
Minimum frequency threshold: 2
Temperature gamma: 20.0
Show N words: 20
Plots to generate: ['betabinom_hellinger']
Figure dimensions: 10.0 x 8.5 inches
Plot directory: Analysis/Most_representative_words/Social_body/
Results directory for CSV files: Analysis/Most_representative_words/Social_body/
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
Using text columns: ['Social_body']
Preprocessing text for each participant (this may take a moment)...
    ⚠️  Text filtered to empty: 'my  back...' -> spaCy tokens: [' ']
    ⚠️  Text filtered to empty: 's...' -> spaCy tokens: ['s']
Grouping word variations...
Using provided transformation dictionary...
Applying groups and converting to presence/absence per participant...
Calculating geometric and probabilistic metrics...

After filtering (freq >= 2): 94 words
Created results directory: Analysis/Most_representative_words/Social_body/

Generating 1 plot(s)...
  CSV results saved to: Analysis/Most_representative_words/Social_body/hellinger_distance_between_betabinomial_distributions_video_21.csv
2 [ 0.4830794  -0.71451226]
3 [ 0.76926863 -0.77467498]
  Plot saved to: Analysis/Most_representative_words/Social_body/hellinger_distance_between_betabinomial_distributions_video_21.pdf

Analysis complete!
Looks like you are using a tranform that doesn't support FancyArrowPatch, using ax.annotate instead. The arrows might strike through texts. Increasing shrinkA in arrowprops might help.
============================================================
Geometric Association Test
============================================================
Input file: Processed Data/touch_data_fixed.psv.txt
Target video(s): [22]
Text columns: ['Social_body']
Minimum frequency threshold: 2
Temperature gamma: 20.0
Show N words: 20
Plots to generate: ['betabinom_hellinger']
Figure dimensions: 10.0 x 8.5 inches
Plot directory: Analysis/Most_representative_words/Social_body/
Results directory for CSV files: Analysis/Most_representative_words/Social_body/
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
Using text columns: ['Social_body']
Preprocessing text for each participant (this may take a moment)...
    ⚠️  Text filtered to empty: 'my  back...' -> spaCy tokens: [' ']
    ⚠️  Text filtered to empty: 's...' -> spaCy tokens: ['s']
Grouping word variations...
Using provided transformation dictionary...
Applying groups and converting to presence/absence per participant...
Calculating geometric and probabilistic metrics...

After filtering (freq >= 2): 94 words
Created results directory: Analysis/Most_representative_words/Social_body/

Generating 1 plot(s)...
  CSV results saved to: Analysis/Most_representative_words/Social_body/hellinger_distance_between_betabinomial_distributions_video_22.csv
13 [-0.0499146  -0.66172702]
14 [0.31759802 0.80099502]
  Plot saved to: Analysis/Most_representative_words/Social_body/hellinger_distance_between_betabinomial_distributions_video_22.pdf

Analysis complete!
Looks like you are using a tranform that doesn't support FancyArrowPatch, using ax.annotate instead. The arrows might strike through texts. Increasing shrinkA in arrowprops might help.
============================================================
Geometric Association Test
============================================================
Input file: Processed Data/touch_data_fixed.psv.txt
Target video(s): [23]
Text columns: ['Social_body']
Minimum frequency threshold: 2
Temperature gamma: 20.0
Show N words: 20
Plots to generate: ['betabinom_hellinger']
Figure dimensions: 10.0 x 8.5 inches
Plot directory: Analysis/Most_representative_words/Social_body/
Results directory for CSV files: Analysis/Most_representative_words/Social_body/
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
Using text columns: ['Social_body']
Preprocessing text for each participant (this may take a moment)...
    ⚠️  Text filtered to empty: 'my  back...' -> spaCy tokens: [' ']
    ⚠️  Text filtered to empty: 's...' -> spaCy tokens: ['s']
Grouping word variations...
Using provided transformation dictionary...
Applying groups and converting to presence/absence per participant...
Calculating geometric and probabilistic metrics...

After filtering (freq >= 2): 94 words
Created results directory: Analysis/Most_representative_words/Social_body/

Generating 1 plot(s)...
  CSV results saved to: Analysis/Most_representative_words/Social_body/hellinger_distance_between_betabinomial_distributions_video_23.csv
9 [-0.11864271  0.15672983]
10 [ 0.86094035 -0.64436679]
  Plot saved to: Analysis/Most_representative_words/Social_body/hellinger_distance_between_betabinomial_distributions_video_23.pdf

Analysis complete!
Looks like you are using a tranform that doesn't support FancyArrowPatch, using ax.annotate instead. The arrows might strike through texts. Increasing shrinkA in arrowprops might help.
============================================================
Geometric Association Test
============================================================
Input file: Processed Data/touch_data_fixed.psv.txt
Target video(s): [24]
Text columns: ['Social_body']
Minimum frequency threshold: 2
Temperature gamma: 20.0
Show N words: 20
Plots to generate: ['betabinom_hellinger']
Figure dimensions: 10.0 x 8.5 inches
Plot directory: Analysis/Most_representative_words/Social_body/
Results directory for CSV files: Analysis/Most_representative_words/Social_body/
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
Using text columns: ['Social_body']
Preprocessing text for each participant (this may take a moment)...
    ⚠️  Text filtered to empty: 'my  back...' -> spaCy tokens: [' ']
    ⚠️  Text filtered to empty: 's...' -> spaCy tokens: ['s']
Grouping word variations...
Using provided transformation dictionary...
Applying groups and converting to presence/absence per participant...
Calculating geometric and probabilistic metrics...

After filtering (freq >= 2): 94 words
Created results directory: Analysis/Most_representative_words/Social_body/

Generating 1 plot(s)...
  CSV results saved to: Analysis/Most_representative_words/Social_body/hellinger_distance_between_betabinomial_distributions_video_24.csv
  Plot saved to: Analysis/Most_representative_words/Social_body/hellinger_distance_between_betabinomial_distributions_video_24.pdf

Analysis complete!
Looks like you are using a tranform that doesn't support FancyArrowPatch, using ax.annotate instead. The arrows might strike through texts. Increasing shrinkA in arrowprops might help.
============================================================
Geometric Association Test
============================================================
Input file: Processed Data/touch_data_fixed.psv.txt
Target video(s): [3, 11, 7, 19, 17, 23, 9, 1, 21, 5, 13, 15]
Text columns: ['Social_body']
Minimum frequency threshold: 2
Temperature gamma: 20.0
Show N words: 20
Plots to generate: ['betabinom_hellinger']
Figure dimensions: 10.0 x 8.5 inches
Plot directory: Analysis/Most_representative_words/Social_body/
Results directory for CSV files: Analysis/Most_representative_words/Social_body/
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
Using text columns: ['Social_body']
Preprocessing text for each participant (this may take a moment)...
    ⚠️  Text filtered to empty: 'my  back...' -> spaCy tokens: [' ']
    ⚠️  Text filtered to empty: 's...' -> spaCy tokens: ['s']
Grouping word variations...
Using provided transformation dictionary...
Applying groups and converting to presence/absence per participant...
Calculating geometric and probabilistic metrics...

After filtering (freq >= 2): 94 words
Created results directory: Analysis/Most_representative_words/Social_body/

Generating 1 plot(s)...
  CSV results saved to: Analysis/Most_representative_words/Social_body/hellinger_distance_between_betabinomial_distributions_video_3_11_7_19_17_23_9_1_21_5_13_15.csv
  Plot saved to: Analysis/Most_representative_words/Social_body/hellinger_distance_between_betabinomial_distributions_video_3_11_7_19_17_23_9_1_21_5_13_15.pdf

Analysis complete!
Looks like you are using a tranform that doesn't support FancyArrowPatch, using ax.annotate instead. The arrows might strike through texts. Increasing shrinkA in arrowprops might help.
============================================================
Geometric Association Test
============================================================
Input file: Processed Data/touch_data_fixed.psv.txt
Target video(s): [16, 22, 14, 12, 4, 2, 10, 8, 24, 6, 18, 20]
Text columns: ['Social_body']
Minimum frequency threshold: 2
Temperature gamma: 20.0
Show N words: 20
Plots to generate: ['betabinom_hellinger']
Figure dimensions: 10.0 x 8.5 inches
Plot directory: Analysis/Most_representative_words/Social_body/
Results directory for CSV files: Analysis/Most_representative_words/Social_body/
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
Using text columns: ['Social_body']
Preprocessing text for each participant (this may take a moment)...
    ⚠️  Text filtered to empty: 'my  back...' -> spaCy tokens: [' ']
    ⚠️  Text filtered to empty: 's...' -> spaCy tokens: ['s']
Grouping word variations...
Using provided transformation dictionary...
Applying groups and converting to presence/absence per participant...
Calculating geometric and probabilistic metrics...

After filtering (freq >= 2): 94 words
Created results directory: Analysis/Most_representative_words/Social_body/

Generating 1 plot(s)...
  CSV results saved to: Analysis/Most_representative_words/Social_body/hellinger_distance_between_betabinomial_distributions_video_16_22_14_12_4_2_10_8_24_6_18_20.csv
  Plot saved to: Analysis/Most_representative_words/Social_body/hellinger_distance_between_betabinomial_distributions_video_16_22_14_12_4_2_10_8_24_6_18_20.pdf

Analysis complete!
Looks like you are using a tranform that doesn't support FancyArrowPatch, using ax.annotate instead. The arrows might strike through texts. Increasing shrinkA in arrowprops might help.
============================================================
Geometric Association Test
============================================================
Input file: Processed Data/touch_data_fixed.psv.txt
Target video(s): [16, 14, 3, 4, 2, 17, 6, 18, 1, 5, 13, 15]
Text columns: ['Social_body']
Minimum frequency threshold: 2
Temperature gamma: 20.0
Show N words: 20
Plots to generate: ['betabinom_hellinger']
Figure dimensions: 10.0 x 8.5 inches
Plot directory: Analysis/Most_representative_words/Social_body/
Results directory for CSV files: Analysis/Most_representative_words/Social_body/
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
Using text columns: ['Social_body']
Preprocessing text for each participant (this may take a moment)...
    ⚠️  Text filtered to empty: 'my  back...' -> spaCy tokens: [' ']
    ⚠️  Text filtered to empty: 's...' -> spaCy tokens: ['s']
Grouping word variations...
Using provided transformation dictionary...
Applying groups and converting to presence/absence per participant...
Calculating geometric and probabilistic metrics...

After filtering (freq >= 2): 94 words
Created results directory: Analysis/Most_representative_words/Social_body/

Generating 1 plot(s)...
  CSV results saved to: Analysis/Most_representative_words/Social_body/hellinger_distance_between_betabinomial_distributions_video_16_14_3_4_2_17_6_18_1_5_13_15.csv
  Plot saved to: Analysis/Most_representative_words/Social_body/hellinger_distance_between_betabinomial_distributions_video_16_14_3_4_2_17_6_18_1_5_13_15.pdf

Analysis complete!
Looks like you are using a tranform that doesn't support FancyArrowPatch, using ax.annotate instead. The arrows might strike through texts. Increasing shrinkA in arrowprops might help.
============================================================
Geometric Association Test
============================================================
Input file: Processed Data/touch_data_fixed.psv.txt
Target video(s): [22, 11, 12, 7, 19, 10, 8, 24, 23, 9, 20, 21]
Text columns: ['Social_body']
Minimum frequency threshold: 2
Temperature gamma: 20.0
Show N words: 20
Plots to generate: ['betabinom_hellinger']
Figure dimensions: 10.0 x 8.5 inches
Plot directory: Analysis/Most_representative_words/Social_body/
Results directory for CSV files: Analysis/Most_representative_words/Social_body/
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
Using text columns: ['Social_body']
Preprocessing text for each participant (this may take a moment)...
    ⚠️  Text filtered to empty: 'my  back...' -> spaCy tokens: [' ']
    ⚠️  Text filtered to empty: 's...' -> spaCy tokens: ['s']
Grouping word variations...
Using provided transformation dictionary...
Applying groups and converting to presence/absence per participant...
Calculating geometric and probabilistic metrics...

After filtering (freq >= 2): 94 words
Created results directory: Analysis/Most_representative_words/Social_body/

Generating 1 plot(s)...
  CSV results saved to: Analysis/Most_representative_words/Social_body/hellinger_distance_between_betabinomial_distributions_video_22_11_12_7_19_10_8_24_23_9_20_21.csv
  Plot saved to: Analysis/Most_representative_words/Social_body/hellinger_distance_between_betabinomial_distributions_video_22_11_12_7_19_10_8_24_23_9_20_21.pdf

Analysis complete!
Looks like you are using a tranform that doesn't support FancyArrowPatch, using ax.annotate instead. The arrows might strike through texts. Increasing shrinkA in arrowprops might help.
============================================================
Geometric Association Test
============================================================
Input file: Processed Data/touch_data_fixed.psv.txt
Target video(s): [16, 22, 14, 19, 17, 24, 18, 23, 20, 21, 13, 15]
Text columns: ['Social_body']
Minimum frequency threshold: 2
Temperature gamma: 20.0
Show N words: 20
Plots to generate: ['betabinom_hellinger']
Figure dimensions: 10.0 x 8.5 inches
Plot directory: Analysis/Most_representative_words/Social_body/
Results directory for CSV files: Analysis/Most_representative_words/Social_body/
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
Using text columns: ['Social_body']
Preprocessing text for each participant (this may take a moment)...
    ⚠️  Text filtered to empty: 'my  back...' -> spaCy tokens: [' ']
    ⚠️  Text filtered to empty: 's...' -> spaCy tokens: ['s']
Grouping word variations...
Using provided transformation dictionary...
Applying groups and converting to presence/absence per participant...
Calculating geometric and probabilistic metrics...

After filtering (freq >= 2): 94 words
Created results directory: Analysis/Most_representative_words/Social_body/

Generating 1 plot(s)...
  CSV results saved to: Analysis/Most_representative_words/Social_body/hellinger_distance_between_betabinomial_distributions_video_16_22_14_19_17_24_18_23_20_21_13_15.csv
  Plot saved to: Analysis/Most_representative_words/Social_body/hellinger_distance_between_betabinomial_distributions_video_16_22_14_19_17_24_18_23_20_21_13_15.pdf

Analysis complete!
Looks like you are using a tranform that doesn't support FancyArrowPatch, using ax.annotate instead. The arrows might strike through texts. Increasing shrinkA in arrowprops might help.
============================================================
Geometric Association Test
============================================================
Input file: Processed Data/touch_data_fixed.psv.txt
Target video(s): [3, 11, 12, 7, 4, 2, 10, 8, 6, 9, 1, 5]
Text columns: ['Social_body']
Minimum frequency threshold: 2
Temperature gamma: 20.0
Show N words: 20
Plots to generate: ['betabinom_hellinger']
Figure dimensions: 10.0 x 8.5 inches
Plot directory: Analysis/Most_representative_words/Social_body/
Results directory for CSV files: Analysis/Most_representative_words/Social_body/
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
Using text columns: ['Social_body']
Preprocessing text for each participant (this may take a moment)...
    ⚠️  Text filtered to empty: 'my  back...' -> spaCy tokens: [' ']
    ⚠️  Text filtered to empty: 's...' -> spaCy tokens: ['s']
Grouping word variations...
Using provided transformation dictionary...
Applying groups and converting to presence/absence per participant...
Calculating geometric and probabilistic metrics...

After filtering (freq >= 2): 94 words
Created results directory: Analysis/Most_representative_words/Social_body/

Generating 1 plot(s)...
  CSV results saved to: Analysis/Most_representative_words/Social_body/hellinger_distance_between_betabinomial_distributions_video_3_11_12_7_4_2_10_8_6_9_1_5.csv
  Plot saved to: Analysis/Most_representative_words/Social_body/hellinger_distance_between_betabinomial_distributions_video_3_11_12_7_4_2_10_8_6_9_1_5.pdf

Analysis complete!
Looks like you are using a tranform that doesn't support FancyArrowPatch, using ax.annotate instead. The arrows might strike through texts. Increasing shrinkA in arrowprops might help.
============================================================
Geometric Association Test
============================================================
Input file: Processed Data/touch_data_fixed.psv.txt
Target video(s): [14, 7, 19, 2, 8, 20, 1, 13]
Text columns: ['Social_body']
Minimum frequency threshold: 2
Temperature gamma: 20.0
Show N words: 20
Plots to generate: ['betabinom_hellinger']
Figure dimensions: 10.0 x 8.5 inches
Plot directory: Analysis/Most_representative_words/Social_body/
Results directory for CSV files: Analysis/Most_representative_words/Social_body/
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
Using text columns: ['Social_body']
Preprocessing text for each participant (this may take a moment)...
    ⚠️  Text filtered to empty: 'my  back...' -> spaCy tokens: [' ']
    ⚠️  Text filtered to empty: 's...' -> spaCy tokens: ['s']
Grouping word variations...
Using provided transformation dictionary...
Applying groups and converting to presence/absence per participant...
Calculating geometric and probabilistic metrics...

After filtering (freq >= 2): 94 words
Created results directory: Analysis/Most_representative_words/Social_body/

Generating 1 plot(s)...
  CSV results saved to: Analysis/Most_representative_words/Social_body/hellinger_distance_between_betabinomial_distributions_video_14_7_19_2_8_20_1_13.csv
32 [0.45356446 0.36985498]
33 [-0.09452802 -0.5639169 ]
  Plot saved to: Analysis/Most_representative_words/Social_body/hellinger_distance_between_betabinomial_distributions_video_14_7_19_2_8_20_1_13.pdf

Analysis complete!
Looks like you are using a tranform that doesn't support FancyArrowPatch, using ax.annotate instead. The arrows might strike through texts. Increasing shrinkA in arrowprops might help.
============================================================
Geometric Association Test
============================================================
Input file: Processed Data/touch_data_fixed.psv.txt
Target video(s): [16, 22, 3, 4, 10, 9, 21, 15]
Text columns: ['Social_body']
Minimum frequency threshold: 2
Temperature gamma: 20.0
Show N words: 20
Plots to generate: ['betabinom_hellinger']
Figure dimensions: 10.0 x 8.5 inches
Plot directory: Analysis/Most_representative_words/Social_body/
Results directory for CSV files: Analysis/Most_representative_words/Social_body/
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
Using text columns: ['Social_body']
Preprocessing text for each participant (this may take a moment)...
    ⚠️  Text filtered to empty: 'my  back...' -> spaCy tokens: [' ']
    ⚠️  Text filtered to empty: 's...' -> spaCy tokens: ['s']
Grouping word variations...
Using provided transformation dictionary...
Applying groups and converting to presence/absence per participant...
Calculating geometric and probabilistic metrics...

After filtering (freq >= 2): 94 words
Created results directory: Analysis/Most_representative_words/Social_body/

Generating 1 plot(s)...
  CSV results saved to: Analysis/Most_representative_words/Social_body/hellinger_distance_between_betabinomial_distributions_video_16_22_3_4_10_9_21_15.csv
  Plot saved to: Analysis/Most_representative_words/Social_body/hellinger_distance_between_betabinomial_distributions_video_16_22_3_4_10_9_21_15.pdf

Analysis complete!
Looks like you are using a tranform that doesn't support FancyArrowPatch, using ax.annotate instead. The arrows might strike through texts. Increasing shrinkA in arrowprops might help.
============================================================
Geometric Association Test
============================================================
Input file: Processed Data/touch_data_fixed.psv.txt
Target video(s): [11, 12, 17, 24, 6, 18, 23, 5]
Text columns: ['Social_body']
Minimum frequency threshold: 2
Temperature gamma: 20.0
Show N words: 20
Plots to generate: ['betabinom_hellinger']
Figure dimensions: 10.0 x 8.5 inches
Plot directory: Analysis/Most_representative_words/Social_body/
Results directory for CSV files: Analysis/Most_representative_words/Social_body/
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
Using text columns: ['Social_body']
Preprocessing text for each participant (this may take a moment)...
    ⚠️  Text filtered to empty: 'my  back...' -> spaCy tokens: [' ']
    ⚠️  Text filtered to empty: 's...' -> spaCy tokens: ['s']
Grouping word variations...
Using provided transformation dictionary...
Applying groups and converting to presence/absence per participant...
Calculating geometric and probabilistic metrics...

After filtering (freq >= 2): 94 words
Created results directory: Analysis/Most_representative_words/Social_body/

Generating 1 plot(s)...
  CSV results saved to: Analysis/Most_representative_words/Social_body/hellinger_distance_between_betabinomial_distributions_video_11_12_17_24_6_18_23_5.csv
37 [ 0.81293605 -0.44615641]
39 [-0.8963542  -0.72801645]
  Plot saved to: Analysis/Most_representative_words/Social_body/hellinger_distance_between_betabinomial_distributions_video_11_12_17_24_6_18_23_5.pdf

Analysis complete!

```


Data aggregation:

```text
Using ['Word', 'Total_Freq'] as ID columns
Using Betabinom_Hellinger as variable columns

Compiled DataFrame:
        Word  ...  videos_11_12_17_24_6_18_23_5
0   shoulder  ...                      0.449782
1       feel  ...                     -0.254694
2       nose  ...                     -0.271473
3      place  ...                     -0.157036
4      crown  ...                      0.217650
..       ...  ...                           ...
89      face  ...                     -0.250660
90       arm  ...                      0.101370
91     upper  ...                      0.075403
92   forearm  ...                      0.128673
93     thigh  ...                     -0.529961

[94 rows x 36 columns]

Saving to CSV: Analysis/Most_representative_words/Social_body/aggregated_data.csv.txt

```