# c10-team-burera

# Training Pipeline

- Training data consisted of balanced subsets from ToxicChat (768), RealToxicityPrompts (536), BeaverTails (200), and Jigsaw Toxic Comment Classification (800), for a total of 2,304 examples.
- The required `google/gemma-2-2b` model was kept frozen and used to extract hidden-state representations.
- Inputs were tokenized with a maximum sequence length of 64 tokens.
- Hidden states were mean-pooled using the attention mask.
- Layer 11 representations were used for the final probe.
- The probe consisted of `StandardScaler` followed by `LogisticRegression`.
- Regularization values of `0.0001`, `0.0003`, `0.001`, `0.003`, and `0.01` were evaluated using stratified 5-fold cross-validation.
- `C=0.001` achieved the best mean cross-validation accuracy (0.8681).
- The final classifier used `C=0.001`, `max_iter=5000`, and `random_state=42`.
- A classification threshold of 0.20 was selected based on development experiments.
- The trained probe and threshold were saved using `joblib`.

# Evaluation

- Stratified 5-fold cross-validation was used to compare probe configurations during development.
- A held-out RealToxicityPrompts set was used to evaluate accuracy, precision, recall, F1, and ROC-AUC.
- Multiple training-data compositions, Gemma layers, regularization values, and classification thresholds were evaluated.
- The final configuration achieved **0.87 accuracy** on the Codabench development leaderboard.

# Reproduction

- The submission consists of a **single notebook** containing the complete training and representation-extraction pipeline.
- The notebook loads the datasets, extracts Gemma hidden-state representations, trains the linear probe, and generates the trained probe artifact.
- The resulting `trained_probe.joblib` and `classifier.py` are packaged according to the competition submission requirements.

# Appendix

## Contributors / Team Members

- Precious Akogun — Team Lead


## Mentors

- Moses Olafenwa
