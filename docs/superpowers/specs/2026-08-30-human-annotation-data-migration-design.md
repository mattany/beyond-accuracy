# Human Annotation Data Migration Design

## Goal

Move the commented human-annotation appendix tables out of the ACL manuscript
and publish their data as a machine-readable artifact in the thesis repository.

## Data artifact

Create `human_study/judge_validation/human_annotation_data.csv` with one row per
question and rhetorical-device metric. Use these columns:

- `question_id`
- `question`
- `metric`
- `model_score`
- `annotator_a`
- `annotator_b`
- `annotator_c`

The normalized layout represents Humor, Connection to Everyday Life, Metaphor,
Scaffolding, and Analogy uniformly. Values and truncated question text will
match the manuscript tables exactly.

## Documentation

Add the CSV to the root README's repository map and human-annotation
documentation, explaining that it contains the complete per-item model scores
and three binary human ratings formerly embedded in the manuscript appendix.

## Source cleanup

Remove the commented appendix block beginning with
`% TODO move to replication materials` from `acl_paper/latex/main.tex`, leaving
the surrounding deployment discussion and `\end{document}` intact.

## Verification and delivery

Validate the CSV schema, row counts, metric counts, score ranges, and binary
annotation columns. Confirm the manuscript still compiles or, if its normal
build dependencies are unavailable, at least verify the edited source boundary.
Commit and push the thesis and ACL repository changes separately without
including unrelated untracked files.
