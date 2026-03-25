# Understanding Attention: A From-Scratch Transformer and Its Learning Dynamics

## Project Goal
This project investigates how attention mechanisms learn and behave under different architectural settings, with a focus on understanding training dynamics and failure modes.

## Task Choice
- Selected task: next-token prediction on a tiny text corpus.
- Reason: simple enough to inspect attention and failure patterns clearly.

## What Is Implemented
- Tokenization and vocabulary creation.
- Fixed-length sequence generation for next-token prediction.
- Transformer encoder components built manually in PyTorch:
  - Token embeddings
  - Sinusoidal positional encoding
  - Scaled dot-product self-attention
  - Multi-head attention
  - Feed-forward network
  - Layer normalization and residual connections
- Output token prediction head.
- Cross-entropy loss and Adam optimization.

## Attention Equation
Attention is computed as:

Attention(Q, K, V) = softmax((QK^T) / sqrt(d_k)) V

## Learning Dynamics Tracked
- Training loss per epoch.
- Gradient norm per epoch (for stability/convergence signal).

## Failure Mode Analysis
The pipeline extracts incorrect predictions with:
- Input sequence
- Mismatch position
- True token vs predicted token
- Model confidence

This helps identify overconfident or context-limited failures.

## Experiment Variations
The script runs 4 experiments:
- Base configuration
- Fewer heads (num_heads = 2)
- Larger embedding size (d_model = 96)
- Longer context window (seq_len = 8)

Outputs are compared in a single JSON summary.

## Visualizations
Generated plots are saved in outputs/:
- Loss + gradient dynamics plot
- Attention heatmap (average across heads)

## Observations From Current Run
- Loss decreases consistently in all runs, indicating convergence.
- Base run final loss: 0.0521
- Heads=2 run final loss: 0.0509
- Embedding=96 run final loss: 0.0175 (best among tested variants)
- Seq length=8 run final loss: 0.0442
- Gradient norms decrease over epochs, suggesting improved optimization stability.

## Failure Cases (Observed)
- In-distribution training failures are near zero after convergence.
- Probe-based out-of-distribution failures are present and intentionally captured.
- Typical failure pattern: unknown-heavy contexts (<unk>) push the model toward frequent tokens or punctuation.
- Example behavior: model predicts "." or common words with moderate-to-high confidence even when target token is unknown.

This demonstrates an important limitation: attention can fit small in-domain data well, but generalization degrades when token distribution shifts.

## Why Attention Works and When It Fails
- Why it works:
   - Self-attention lets each token aggregate context from all positions in the window.
   - Multi-head structure captures multiple relationship types (local and broader token interactions).
   - Residual connections and layer normalization improve trainability and stable gradient flow.
- When it fails:
   - Tiny datasets produce narrow token coverage and overfitting to frequent patterns.
   - Unknown tokens collapse diverse semantics into <unk>, reducing meaningful key/query matching.
   - Fixed context window limits long-range dependency handling.

## How To Run
1. Install dependencies:
   pip install -r requirements.txt
2. Run project:
   python src/transformer_project.py

## Output Files
- outputs/base_results.json
- outputs/heads_2_results.json
- outputs/embed_96_results.json
- outputs/seq_8_results.json
- outputs/experiment_comparison.json
- outputs/*_loss_grad.png
- outputs/*_attention_heatmap.png
