# April 2026 Mech Interp Puzzle: Max of List

- solution by Dan Wilhelm [dan@danwil.com]

Herein we provide solutions (max_of_list.ipynb) for the [April 2026 puzzle: Max of List](https://puzzles.baulab.info/april-2026.html).

- We provide a cross-platform custom NumPy implementation and verify its equivalency.
- We focus on showing the max() algorithm main path, rather than corner-cases and EOS production.

---
## Table of Contents

- Setup code: Loads helper modules.
- Sidebar: We show counterexamples that Puzzle 2 is not 100% accurate. We provide potential explanations in Appendix 3-3.
- Sampling code: Samples activations from each model.

### Puzzle 1
We introduce three experimental techniques that take advantage of the model's linearity (i.e. absence of layer norms), using them to solve Puzzle 1. We show that attention is sufficiently powerful to immediately find the maximum digit.

1. (1-1) We analyze layer 0 attention scores via `S[E + P] = S[E] + S[P]` (E=embeds, P=positional embeds).
    - Using this, we show how heads independently use E and P to take the max, copy tens digit information, and copy previously-copied information.
2. (1-2) We use logistic regression on head residual contributions (i.e. `W_V[hi] @ W_O[hi]`) to determine whether each head "knows" the true class.
    - We show head 3 places the max digit on the residual stream, alongside mostly per-class constants from the other heads.
3. (1-3) We analyze layer 1 attention head logit attributions by partitioning W_O, e.g. for head hi: `W_V[hi] @ W_O[hi] @ W_U.T`.
    - We show that each head either outputs info about the true class`*`, or contributes a surprisingly uniform per-class bias.
    - Specifically, heads 0&2 reinforce head 3 in knowing and promoting only true classes 7-9 (which together are the maxes for >67% of all possible sequences).

### Puzzle 2

- (2-1) We first broadly visualize the below algorithm using a grid of logistic regression results.

Tens digit:
1. (2-2) Layer 0 head 1 identifies the max tens digit as in Puzzle 1, copying it to the ANS residual row.
2. (2-5) The max tens digit is already known and on the residual stream.
    - Hence, layer 1 heads (A) reinforce the true class (especially heads 1&2) and (B) provide per-class constants to properly match the unembeddings.

Ones digit:
1. (2-3) Layer 0 heads 0&2: Attend to the ANS row, which we know from above contains the tens digit. This info is copied to the ANS+1 residual row.
2. (2-4) Layer 0 head 3: Info about each tens digit is copied into the corresponding ones-digit activation rows.
3. (2-6) Layer 1 heads 0&3: Recognize the max ones digit by (A) matching each (2-4) with (2-3) for a score boost, (B) applying a score gradient to the digits themselves, and (C) applying a mask on the ones digits via the positional embeds.

### Appendices
1. (3-1) Ablation experiment results, particularly showing contribution of head 3 (puzzle 1) and puzzle 2 head combinations.
2. (3-2) Attention pattern explorer.

`*` We refer to "output class" as a possible output token, and "true class" as the *correct* token.