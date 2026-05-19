# Social Touch – Prolific: Survey Flow Logic

This document describes the branching, looping, and conditional display logic of the Qualtrics survey (`Social_Touch_-_Prolific.qsf`). It is intended to supplement the human-readable question list in `Social_Touch_-_Prolific.md`.

---

## Top-level flow

```
START
 │
 ├─ [SET] Capture PROLIFIC_PID from Prolific URL parameter
 │
 ├─ [BLOCK] Welcome
 │
 ├─ [BRANCH] Device type is mobile?
 │    └─ YES → [SET] Device = "Mobile"   (recorded in embedded data; no routing effect)
 │
 ├─ [BLOCK] Informed Consent
 │
 ├─ [BLOCK] ProlificID   (auto-filled from URL; collected before consent check)
 │
 ├─ [BRANCH] Consent = "I do not want to participate"?
 │    └─ YES → [END] Display screen-out message  ──────────────────────── EXIT
 │
 ├─ [BLOCK] Demographics
 │
 ├─ [SET] Current_loop = current loop iteration number
 │
 ├─ [BLOCK] Video  ← looping block (see below)
 │
 └─ [END] Redirect to Prolific completion URL
         (https://app.prolific.co/submissions/complete?cc=C17HNO1D)
```

---

## Video block looping

The **Video** block uses Qualtrics Question Looping. It loops over the choices displayed by the hidden question `Q359` (export tag), which holds the 24 video stimulus URLs.

- Qualtrics randomly selects a subset of those 24 videos and presents them one per loop iteration.
- The variable `Current_loop` records which iteration is currently running.
- The loop continues or stops based on the participant's answer to the **Continue** question at the end of each iteration (see below).

```
VIDEO BLOCK (one iteration per video stimulus, random order)
 │
 ├─ [DISPLAY] "Please focus on the video shown"  (video embedded via loop URL)
 │
 ├─ [DISPLAY] Social scenario instruction
 │
 ├─ [TEXT] Social_self_02   — Who are you with?
 ├─ [TEXT] Social_body_02   — What part of your body is touched?
 ├─ [TEXT] Social_place_02  — Where are you?
 ├─ [TEXT] Social_context   — What is happening?
 ├─ [TEXT] Intention&Purpose_02 — What is the intention/purpose?
 │
 ├─ [MC]   Appropriateness  — Would this touch be appropriate? (single-answer)
 │
 ├─ [DISPLAY] "Here is the same video again…"
 │
 ├─ [TEXT] Sensory_02       — Physical characteristics of the touch
 ├─ [HM]  Valence&Arousal   — Click on valence × arousal grid
 ├─ [TEXT] Emotional_self   — How does this touch make you feel emotionally?
 ├─ [TEXT] Q373             — Additional touch descriptor (open text)
 │
 ├─ [MC]   Continue         — "Would you be willing to watch another video?"
 │    ├─ "YES, show me another video"  → loop back to next video stimulus
 │    └─ "NO, I would like to finish"  → exit loop
 │
 └─ [DISPLAY] Input  — "Thank you…" + Prolific redirect button
      └─ SHOWN ONLY IF: Continue = "NO, I would like to finish"
```

> **Note on the numbered video blocks** (2nd video – 24th video): these blocks exist in the survey file but are **not part of the active flow**. They appear to be a legacy structure from an earlier version of the survey, replaced by the looping Video block above.

---

## Conditional display logic

| Question | Export tag | Shown only if |
|---|---|---|
| "Thank you…" / Prolific redirect | `Input` | Continue (`QID369`) = "NO, I would like to finish the survey now." |
| "Where do you feel you belong?" | `Q367` | Demographics question `Q365` (`QID738`) = "Yes" |

---

## Embedded data variables

| Variable | Set when | Value |
|---|---|---|
| `PROLIFIC_PID` | Survey start | Participant's Prolific ID (from URL parameter) |
| `Device` | If device is mobile | `"Mobile"` |
| `Current_loop` | Before each video iteration | Current loop iteration number |

---

## Completion paths

| Path | Trigger | Outcome |
|---|---|---|
| Screen-out | Consent = "I do not want to participate" | End survey with message; no Prolific credit |
| Normal completion | All video loops done, or participant declines to continue | Redirect to Prolific completion URL |
