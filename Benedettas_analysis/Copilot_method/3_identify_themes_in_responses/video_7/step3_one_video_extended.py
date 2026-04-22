# step3_one_video.py
# Purpose: Step 3 for ONE video (an Excel with responses).
# Loads per-theme wordsets from results/<theme>/wordnet_selections_synonyms.json,
# evaluates each response, and outputs detailed + summary CSVs.

import os
import json
import re
import csv
import argparse
from collections import defaultdict, Counter
from typing import Dict, List, Set, Tuple, Iterable

# ---------- Excel reading (pandas, openpyxl) ----------
# toolkit for working with data in tables — like a super-powered spreadsheet inside Python.
import pandas as pd

# ---------- Config knobs ----------
# weight is a varible
WEIGHTS = {"original": 1.0, "synonym": 0.75,
           "closest": 0.50}  # weight is a dictionary
# threshold for theme presence, sets a minimum score needed for a theme to be considered "present" in the text.
DEFAULT_THRESHOLD = 1.0
# max_ngram is a variable #will look at phrases up to 4 words long. So it checks single words, 2-word phrases, 3-word phrases, and 4-word phrases — but nothing longer.
MAX_NGRAM = 4
NEGATORS = {"no", "not", "never", "none", "n't", "dont", "don't", "do", "do not",  # checks for negators words newrby and if the negator is present that word is not counted
            # set — basically a collection of unique items with no duplicates.
            "without", "hardly", "scarcely", "barely", "neither", "nor"}
# this is telling to look at the three worrds before a word to check for negators
WINDOW_FOR_NEG = 3
# regular expression — a way of describing a pattern of text. Think of it like a search filter. Instead of searching for a specific word like "happy", you're describing a shape of text — like "give me anything made of letters."
WORD_RE = re.compile(r"[a-zA-Z']+")

# ---------- Lemmatizer (lightweight, with fallback) ----------
try:  # ry running this code... but if something goes wrong, don't crash — just do something else instead
    # nltk stands for Natural Language Toolkit — a library built for working with human language text.
    from nltk.stem import WordNetLemmatizer
    _lemm = WordNetLemmatizer()  # lemmatizing is just reducing a word to its base form
# creates the lemmatizer tool and stores it in a variable called _lemm

    def _lem(t, pos):  # t = token to lemmatize; pos = part of speech like verb, noun etc --> tries to lemmatise the word and if that fails it just resturns the word unchnaged

        try:   #
            return _lemm.lemmatize(t, pos)  # try to lemmatize the word
        except Exception:
            return t  # # if it fails, just return the word unchanged
except Exception:  # if it fails, do THIS instead of crashing
    def _lem(t, pos): return t

# ---------- Text helpers ----------

# t: str means "the input t should be a string" and -> str means "this function will return a string." These are called type hints


def norm(t: str) -> str:
    # .lower() converts everything to lowercase — "HELLO" becomes "hello", "Hello" becomes "hello".
    return t.lower().strip()
# strip removes any spaces form the start and end of a string


# takes a piece of text as input, finds every word in that text using WORD_RE, normalises each word with norm() lowercase no extraspaces, returns a clea list of words
def tokenize(text: str) -> List[str]:
    return [norm(m.group()) for m in WORD_RE.finditer(text or "")]


# t lemmatizes the same word four different ways — as a noun, verb, adjective, and adverb — because the base form can differ depending on the part of speech.
def lemmas_for(token: str) -> Set[str]:
    return {
        norm(_lem(token, 'n')),  # n = noun
        norm(_lem(token, 'v')),  # v = verb
        norm(_lem(token, 'a')),  # a = adjective
        norm(_lem(token, 'r')),  # r = adverb
        norm(token)
    }
# collects all possible base forms into a set, making sure nothing gets missed.


def phrase_variants(s: str) -> Set[str]:
    # function first normalises the phrase with norm() — lowercase, no extra spaces
    s = norm(s)
    # it checks: does the phrase contain an underscore?
    return {s, s.replace("_", " ")} if "_" in s else {s}
# If yes — it returns both versions:
# "good_feeling" — with the underscore
# "good feeling" — with a space
# If no — it just returns the phrase as-is.


# Slide a window of n words across the token list and collect every phrase you find.
def ngrams(tokens: List[str], n: int) -> List[str]:
    # for example, if tokens = ["the", "cat", "sat"] and n=2, it will produce: the cat
    return [" ".join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

# ---------- Load theme lexicons from Step 2 ----------

# THIS IS THE FUCNTION TO BE CHNAGED TO TEKLL WHICH FILE TO READ AND USE


def load_theme_lexicon(theme_dir: str) -> Dict[str, Set[str]]:
    path = os.path.join(theme_dir, "wordnet_selections_extended.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")

    originals, synonyms, closest = set(), set(), set()

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            word = (row.get("word") or "").strip()
            syns = (row.get("synonyms") or "").strip()
            # closest words column — new addition
            close = (row.get("closest_words") or "").strip()

            if word and word != "N/A":
                for v in phrase_variants(word):
                    originals.add(v)
                if " " not in word and "_" not in word:
                    originals |= lemmas_for(word)

            if syns and syns != "N/A":
                for syn in syns.split(","):
                    syn = syn.strip()
                    for v in phrase_variants(syn):
                        synonyms.add(v)
                    if " " not in syn and "_" not in syn:
                        synonyms |= lemmas_for(syn)

            if close and close != "N/A":  # same logic as synonyms but for closest words
                for c in close.split(", "):
                    c = c.strip()
                    for v in phrase_variants(c):
                        closest.add(v)
                    if " " not in c and "_" not in c:
                        closest |= lemmas_for(c)

    return {"original": originals, "synonym": synonyms, "closest": closest}


def load_all_themes(results_root: str,  # takes a folder path and an optional list of theme names to filter by. only_themes = None means by default it loads all themes.
                    only_themes: Iterable[str] = None) -> Dict[str, Dict[str, Set[str]]]:
    themes = {}  # empty dictionary that will store all the themes found.
    names_filter = None if not only_themes else {  # If only_themes was provided, it builds a set of theme names to look for. If not, names_filter stays None meaning no filter — load everything.
        n.strip() for n in only_themes}
    # os.listdir lists everything inside a folder — like opening a folder on your computer and seeing all the files and subfolders inside.
    for name in os.listdir(results_root):
        # Builds the full path and checks if it's a folder. If it's not a folder, continue skips it and moves to the next item.
        tdir = os.path.join(results_root, name)
        if not os.path.isdir(tdir):
            continue
        # If a filter was provided and this theme isn't in it — skip it.
        if names_filter and name not in names_filter:
            continue
        try:
            # Tries to load the theme lexicon from the folder. If the JSON file is missing, pass just silently skips it — no crash.
            themes[name] = load_theme_lexicon(tdir)
        except FileNotFoundError:
            # skip folders not processed yet
            pass
    if not themes:  # If after all that, no themes were loaded at all — throw an error with a helpful message.
        raise RuntimeError(
            "No theme lexicons found under results_root (or filter excluded all).")
    return themes  # Returns all loaded themes as a dictionary.

# ---------- Build indexes for a response ----------


def build_response_indexes(text: str) -> Tuple[List[str], Set[str], Set[str]]:
    # takes a text and returns a clean list of lowercase words. So this just chops the input text into a list of words.
    tokens = tokenize(text)
    lemma_set = set()  # these three lines: Goes through every token and collects all its lemma forms — noun, verb, adjective, adverb versions. |= merges them all into one big set. So by the end, lemma_set contains every possible root form of every word in the text.
    for t in tokens:
        lemma_set |= lemmas_for(t)
    # these three lines: loops from 2 to 4 and builds all possible 2-word, 3-word, and 4-word phrases from the tokens. All collected into phrase_set.
    phrase_set = set()
    for n in range(2, MAX_NGRAM+1):
        # For every phrase in phrase_set, it also adds an underscore version. So "good feeling" gets "good_feeling" added too. You saw this logic before with phrase_variants
        phrase_set |= set(ngrams(tokens, n))
    phrase_set |= {p.replace(" ", "_") for p in phrase_set}
    # Returns all three things — the word list, the lemma set, and the phrase set.
    return tokens, lemma_set, phrase_set


# ---------- Matching, negation, scoring ----------


def find_matches(lex: Dict[str, Set[str]],  # def. This function takes the lexicon and the three things built by build_response_indexes. It starts with an empty dictionary with three empty lists — one for each category of match.
                 tokens: List[str],
                 lemma_set: Set[str],
                 phrase_set: Set[str]) -> Dict[str, List[str]]:
    matches = {"original": [], "synonym": [], "closest": []}


# This is a function inside a function — called a nested function. It loops through every word/phrase in the vocabulary and checks two things:
# If the term is a phrase (has a space or underscore) → check if it's in phrase_set
# If it's a single word → check if it's in lemma_set

    def match_bucket(bucket, vocab):
        for term in vocab:
            if (" " in term) or ("_" in term):
                if term in phrase_set or term.replace("_", " ") in phrase_set:
                    matches[bucket].append(term)
            else:
                if term in lemma_set:
                    matches[bucket].append(term)
    # Runs match_bucket three times — once for each category. So it checks originals, then synonyms, then closest words.
    for b in ("original", "synonym", "closest"):
        match_bucket(b, lex[b])
    # Returns the dictionary with all found matches sorted into their three buckets.
    return matches


# def. This function takes the token list and the matches found. Its job is to remove any matches that are negated —
def apply_negation(tokens: List[str], matches: Dict[str, List[str]]) -> Dict[str, List[str]]:
    # defaultdict is a special dictionary that automatically creates an empty list for any new key. So you don't get an error if you look up a word that hasn't been added yet — it just gives you an empty list.
    positions = defaultdict(list)
    for i, t in enumerate(tokens):
        positions[t].append(i)
        # enumerate loops through the tokens and gives you both the position and the word at the same time. So for ["i", "feel", "not", "happy"]: position 0 → "i"
# position 1 → "feel"
# position 2 → "not"
# position 3 → "happy"
# so positions ends up being a dictionary mapping each word to all the positions it appears at in the text


# lines below: takes a term and returns a bool — remember True or False.
# First check — if the term is a phrase (has a space or underscore), it immediately returns False meaning "not negated". Why? Because negation checking only works on single words — it's too complex to check phrases.

    def negated(term: str) -> bool:
        if " " in term or "_" in term:
            return False
        # For every position the word appears at, it looks at the WINDOW_FOR_NEG = 3 words to the left.
        for pos in positions.get(term, []):
            left = tokens[max(0, pos-WINDOW_FOR_NEG):pos]
            if any(t in NEGATORS for t in left):
                return True  # If any of those words is in NEGATORS — return True meaning "yes, this word is negated."
            return False  # If no negator found anywhere, return False.
    # Builds a new dictionary — same three buckets but with negated words removed.
    out = {}
    for b, terms in matches.items():
        out[b] = [t for t in terms if (" " in t or "_" in t) or not negated(t)]
    # keeps a term if it's either a phrase OR if it's not negated. Returns the cleaned matches.
    return out


def score(matches: Dict[str, List[str]], use_closest=True) -> Tuple[float, Counter]:
    # Counter is a special dictionary that counts things. Here it counts how many matches are in each bucket — how many originals, how many synonyms, how many closest words.
    counts = Counter({k: len(v) for k, v in matches.items()})
    s = counts["original"]*WEIGHTS["original"] + \
        counts["synonym"]*WEIGHTS["synonym"]  # Here it multiplies each count by its weight and adds them up. So 2 originals and 1 synonym gives:
# 2 × 1.0 + 1 × 0.75 = 2.75 The \ at the end just means the line continues on the next line — nothing special!
    if use_closest:  # If use_closest is True, it adds the closest word scores too. If not, it sets closest count to 0 and ignores them.
        s += counts["closest"]*WEIGHTS["closest"]
    else:
        counts["closest"] = 0
    # Returns two things — the final score s and the counts breakdown.
    return s, counts


def detect_for_response(text: str,  # def. This function takes the text to analyse, all the loaded themes, a threshold, and whether to use closest words. It's the function that does the full theme detection!
                        themes: Dict[str, Dict[str, Set[str]]],
                        threshold: float = DEFAULT_THRESHOLD,
                        use_closest: bool = True):
    # builds the three search indexes from the text. One line, three things returned!
    tokens, lemma_set, phrase_set = build_response_indexes(text)
    # Empty list to store results. Then loops through every theme one by one.
    results = []
    for name, lex in themes.items():
        m = find_matches(lex, tokens, lemma_set, phrase_set)
        m = apply_negation(tokens, m)
        # for each theme it runs the full pipeline: find matches, remove ngated words, calculate the score
        s, cnt = score(m, use_closest=use_closest)
        results.append({
            "theme": name,
            "present": s >= threshold,
            "score": round(s, 3),
            "counts": dict(cnt),
            "examples": {k: sorted(set(v))[:10] for k, v in m.items()}
            # builds a result dictionary for each theme with five things: theme, present (treu or false), score (rounded to 3 decimals), counts (how many originals, synonyms, closets), examples (up to 10 example words that matched )
        })
    # Sorts all results by score — highest first. lambda is just a quick one-line function that says "sort by the score field".
    results.sort(key=lambda r: r["score"], reverse=True)
    return results  # Returns the full list of theme results.

# ---------- Excel helpers ----------


def read_excel_responses(xlsx_path: str,  # This function takes four things: 1) xlsx_path — the path to the Excel file, 2)sheet — which sheet inside the Excel file to read, 3) no_header — whether the file has a header row or not, 4)text_column — which column contains the text to analyse
                         sheet,
                         no_header: bool,
                         text_column):
    # If no_header is True, header is set to None meaning "there's no header row". Otherwise it's set to 0 meaning "the first row is the header".
    header = None if no_header else 0
    df = pd.read_excel(xlsx_path, sheet_name=sheet,  # reads the Excel file and stores it as a DataFrame — think of a DataFrame like a table with rows and columns, just like a spreadsheet.
                       # sheet_name=sheet — tells it which sheet to open, engine="openpyxl" — the tool pandas uses under the hood to read Excel files, header=header — uses the value we set above
                       engine="openpyxl", header=header)

    # decide which column to use
    if text_column is None:
        # pick the first column that has any non-empty text; If no column was specified, the program tries to find one automatically. It loops through every column, converts it to text, strips spaces, and replaces "nan" (empty cells) with blank. If the column has any non-empty values — use that one and stop looking.
        for col in df.columns:
            series = (df[col].astype(str).str.strip().replace("nan", ""))
            if series.any():
                text_column = col
                break
    else:  # If a column was specified — check what type it is. If it's a string like "0" or "2", convert it to an integer first. Then if it's an integer, use it as a position to get the column name — so 0 means first column, 1 means second column etc.
        # allow numeric index, e.g., 0
        if isinstance(text_column, str) and text_column.isdigit():
            text_column = int(text_column)
        if isinstance(text_column, int):
            text_column = df.columns[text_column]

    # If after all that we still don't have a column — throw an error with a helpful message.
    if text_column is None:
        raise ValueError(
            "Could not detect the text column. Use --text-column to specify.")

    responses = (  # Grabs the text column, converts everything to strings, strips spaces, removes empty cells. Then converts it to a plain Python list — removing any empty strings. Returns the clean list of responses.
        df[text_column].astype(str).str.strip().replace({"nan": ""})
    )
    responses = [r for r in responses.tolist() if r]
    return responses

# ---------- Main (one video) ----------

# Each add_argument defines one option the user can pass in when running the program. Think of them like settings:
# -results-root — where the theme folders are
# -excel-file — which Excel file to analyse
# -sheet — which sheet to use
# -threshold — minimum score to count a theme as present
# -out-detailed — where to save the detailed results


def main():  # argparse is a library for handling command line arguments — settings you pass to the program when you run it from the terminal. ArgumentParser sets it up.
    ap = argparse.ArgumentParser(
        "STEP 3 (one video): detect themes for each response in an Excel file")
    ap.add_argument("--results-root", type=str, required=True,
                    help="Folder with per-theme subfolders that contain wordnet_selections_synonyms.json")
    ap.add_argument("--excel-file", type=str, required=True,
                    help="Path to the single-video Excel (responses)")
    ap.add_argument("--sheet", type=str, default="Sheet1",
                    help="Sheet name or index (default: Sheet1)")
    ap.add_argument("--no-header", action="store_true",
                    help="Specify when the sheet has no header row (first row is data)")
    ap.add_argument("--text-column", type=str,
                    help="Column name or 0-based index (optional; auto-detected if omitted)")
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                    help="Score >= threshold => theme present (default 1.0)")
    ap.add_argument("--no-closest", action="store_true",
                    help="Ignore closest-words expansion")
    ap.add_argument("--only-themes", nargs="*",
                    help="Optional list of theme folder names to include (e.g., theme_tv_chilling romantic)")
    ap.add_argument("--out-detailed", type=str, default="step3_detailed.csv",
                    help="Output CSV: one row per (response × theme)")
    ap.add_argument("--out-summary", type=str, default="step3_summary.csv",
                    help="Output CSV: one row per theme with counts")
    # This reads whatever the user typed in the terminal and stores all the values in args. So args.excel_file gives you the Excel path, args.threshold gives you the threshold etc.
    args = ap.parse_args()

    # Load themes (optionally filtered)-->  loads all the theme lexicons from the folder the user specified.
    themes = load_all_themes(args.results_root, only_themes=args.only_themes)

    # Read responses from Excel
    # not strictly necessary, but allows numeric sheet index
    # Converts the sheet to a number if needed, then calls read_excel_responses — which you know loads and cleans the Excel data.
    sheet_arg = int(args.sheet) if str(args.sheet).isdigit() else args.sheet
    responses = read_excel_responses(
        xlsx_path=args.excel_file,
        sheet=sheet_arg,
        no_header=args.no_header,
        text_column=args.text_column
    )

    # Evaluate
    # Two empty containers to store results. rows will hold every result as a row for the CSV file. present_counter counts how many responses each theme was present in. You saw Counter earlier!
    rows = []
    present_counter = Counter()
    # Loops through every response. enumerate(..., 1) gives both the position and the text — starting from 1 instead of 0. So i=1 for the first response, i=2 for the second etc.
    for i, text in enumerate(responses, 1):
        res = detect_for_response(  # important one
            text,
            themes,
            threshold=args.threshold,
            use_closest=(not args.no_closest)
        )
        # store detailed rows --> master function that runs the full pipeline for one piece of text: Tokenizes it, Finds matches, Applies negation, Calculates scores, Returns results for every theme
        for r in res:
            rows.append({  # Loops through every theme result from detect_for_response. For each one, builds a dictionary and adds it to rows.
                "response_id": i,
                "response": text,
                "theme": r["theme"],
                "present": r["present"],
                "score": r["score"],
                "count_original": r["counts"]["original"],
                "count_synonym": r["counts"]["synonym"],
                "count_closest": r["counts"]["closest"],
                "examples_original": ", ".join(r["examples"].get("original", [])),
                "examples_synonym": ", ".join(r["examples"].get("synonym", [])),
                "examples_closest": ", ".join(r["examples"].get("closest", [])),
                # Each key becomes a column in the final CSV file. Notice ", ".join(...) — that takes a list like ["happy", "joy"] and joins it into a single string "happy, joy" so it fits neatly in a CSV cell. So for every response + theme combination, it records: Which response it was (response_id), The actual text (response), The theme name, Whether it was present, The score, How many originals, synonyms, closest words matched, Example words that matched
            })
            if r["present"]:
                # If the theme was present in this response, add 1 to that theme's counter. So by the end, present_counter tells you how many responses each theme appeared in.
                present_counter[r["theme"]] += 1

    # Write detailed CSV
    if rows:  # Simple check — if there are any rows at all. If no responses were detected, skip saving.
        with open(args.out_detailed, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            # with open(...) before — it safely opens a file. This time "w" means write instead of read., csv.DictWriter — a tool for writing dictionaries as CSV rows, fieldnames=list(rows[0].keys()) — uses the keys from the first row as column headers, w.writeheader() — writes the column names as the first row, w.writerows(rows) — writes all the data rows
            w.writerows(rows)
        # Just prints a message confirming where the file was saved. The f"..." is an f-string. It inserts the actual file path into the message.
        print(f"Saved detailed results to: {args.out_detailed}")
    else:  # If rows was empty — just print a message explaining nothing was saved.
        print("No responses detected in the Excel (detailed CSV not created).")

    # Write summary CSV (per theme)
    if themes:  # if any themes were loaded at all. Just a safety check before trying to write the summary.
        # get all theme names to ensure zeros included
        # Gets all theme names and sorts them alphabetically. The important word here is sorted — it makes sure even themes with zero matches still appear in the summary file.
        all_names = sorted(themes.keys())
        # safely opens the summary CSV file for writing.
        with open(args.out_summary, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            # Writes the header row — just two columns: the theme name and how many responses it appeared in.
            w.writerow(["theme", "responses_with_theme"])
            for name in all_names:
                # Loops through every theme and writes one row each. present_counter.get(name, 0) gets the count for that theme — if it never appeared, it returns 0 instead of crashing.
                w.writerow([name, present_counter.get(name, 0)])
        # Confirms where the summary file was saved.
        print(f"Saved summary counts to: {args.out_summary}")

    # Print a quick console summary
    # prints a header to the terminal.
    print("\n=== STEP 3 SUMMARY (one video) ===")
    for name in sorted(themes.keys()):
        print(f"{name:>22}: {present_counter.get(name, 0)} responses flagged")
# Loops through every theme alphabetically and prints one line per theme. You know `present_counter.get(name, 0)` — gets the count or 0 if not found.
# {name:>22}` — the `:>22` part means *"right-align this text in a space 22 characters wide."* It just makes the output look neat and lined up in the terminal.


if __name__ == "__main__":  # This checks — "is this script being run directly by the user?" Because Python scripts can either be run directly OR imported by another script. If another script imports this one, you don't want main() to run automatically. So this line is basically saying:"Only run the following code if someone ran THIS file directly — not if another script imported it."
    main()
    # If yes — run main()! Which kicks off the whole program.
