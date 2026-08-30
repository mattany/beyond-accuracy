# Human Annotation Data Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the commented ACL appendix annotation tables with a documented, machine-readable CSV in the thesis publication repository.

**Architecture:** Represent every question-device judgment as one normalized CSV row so all five rhetorical-device metrics share one schema. Document the artifact in the thesis root README, then remove only the obsolete commented appendix block from the ACL manuscript.

**Tech Stack:** CSV, Markdown, LaTeX, Python 3 standard library, Git.

## Global Constraints

- Preserve the manuscript values and truncated question text exactly.
- Use columns `question_id,question,metric,model_score,annotator_a,annotator_b,annotator_c`.
- Include exactly 30 rows for each of Humor, Connection to Everyday Life, Metaphor, Scaffolding, and Analogy.
- Do not stage or commit unrelated untracked files in either repository.
- Commit and push each repository separately.

---

### Task 1: Publish and document the normalized annotation CSV

**Files:**
- Create: `human_study/judge_validation/human_annotation_data.csv`
- Modify: `README.md`

**Interfaces:**
- Consumes: The five metric tables in `../acl_paper/latex/main.tex`, beginning at the `% TODO move to replication materials` marker.
- Produces: A normalized CSV with 150 rows and a root README reference to that artifact.

- [ ] **Step 1: Fast-forward thesis main**

Run:

```bash
git -C /Users/mattan.yeroushalmi/studies/thesis pull --ff-only
```

Expected: `main` is up to date with `origin/main`; unrelated untracked files remain untouched.

- [ ] **Step 2: Create the normalized CSV**

Read the three commented LaTeX tables and write:

```csv
question_id,question,metric,model_score,annotator_a,annotator_b,annotator_c
```

For the Humor/Connection and Metaphor/Scaffolding tables, emit two rows per source table row. For the Analogy table, emit one row per source row. Use metric names exactly:

```text
Humor
Connection to Everyday Life
Metaphor
Scaffolding
Analogy
```

Parse numeric model scores as the two-decimal strings shown in the manuscript and preserve each binary annotator value as `0` or `1`. Use Python's `csv.writer` so apostrophes, commas, and quotation marks are escaped correctly.

- [ ] **Step 3: Validate the CSV**

Run:

```bash
python - <<'PY'
import csv
from collections import Counter
from pathlib import Path

path = Path("human_study/judge_validation/human_annotation_data.csv")
with path.open(newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))

expected_fields = [
    "question_id", "question", "metric", "model_score",
    "annotator_a", "annotator_b", "annotator_c",
]
assert rows and list(rows[0]) == expected_fields
assert len(rows) == 150
assert Counter(row["metric"] for row in rows) == {
    "Humor": 30,
    "Connection to Everyday Life": 30,
    "Metaphor": 30,
    "Scaffolding": 30,
    "Analogy": 30,
}
assert all(0.0 <= float(row["model_score"]) <= 1.0 for row in rows)
assert all(
    row[column] in {"0", "1"}
    for row in rows
    for column in ("annotator_a", "annotator_b", "annotator_c")
)
print("validated 150 annotation rows")
PY
```

Expected: `validated 150 annotation rows`.

- [ ] **Step 4: Document the artifact**

Add a repository-map row for `human_study/judge_validation/human_annotation_data.csv`. In the human-study data section, state that the CSV contains the complete per-item LLM scores and binary ratings from three annotators for the five validated rhetorical devices, formerly presented as appendix tables.

- [ ] **Step 5: Commit and push the thesis changes**

Run:

```bash
git add README.md human_study/judge_validation/human_annotation_data.csv
git commit -m "Publish human annotation data"
git push origin main
git status --short --branch
```

Expected: the commit contains only the README and CSV; `main` is aligned with `origin/main`. Unrelated untracked files may still be listed.

---

### Task 2: Remove the migrated ACL appendix block

**Files:**
- Modify: `/Users/mattan.yeroushalmi/studies/acl_paper/latex/main.tex`

**Interfaces:**
- Consumes: The now-published thesis CSV from Task 1.
- Produces: An ACL manuscript ending directly after the deployment appendix prose and `\end{document}`.

- [ ] **Step 1: Remove the exact commented block**

Delete from:

```latex
% TODO move to replication materials
```

through:

```latex
% \end{table*}
```

immediately before `\end{document}`. Preserve the blank-line separation and do not alter the preceding deployment paragraph.

- [ ] **Step 2: Verify source boundaries**

Run:

```bash
python - <<'PY'
from pathlib import Path

text = Path("latex/main.tex").read_text(encoding="utf-8")
assert "% TODO move to replication materials" not in text
assert "% \\label{tab:annotations_humor_conn}" not in text
assert "% \\label{tab:annotations_metaphor_scaff}" not in text
assert "% \\label{tab:annotations_analogy}" not in text
assert text.rstrip().endswith(r"\end{document}")
print("annotation appendix removed cleanly")
PY
```

Expected: `annotation appendix removed cleanly`.

- [ ] **Step 3: Build the manuscript**

Use the repository's established LaTeX build command if available. If no build command or required TeX dependency is available, record that limitation and rely on the focused boundary verification from Step 2.

- [ ] **Step 4: Commit and push the ACL change**

Run:

```bash
git add latex/main.tex
git commit -m "Move annotation tables to replication materials"
git push origin master
git status --short --branch
```

Expected: the commit contains only `latex/main.tex`; `master` is aligned with `origin/master`. Existing unrelated untracked files remain uncommitted.
