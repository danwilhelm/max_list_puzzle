# April 2026 Mech Interp Puzzle: Max of List

by Dan Wilhelm [dan@danwil.com]

Herein we provide solutions for the [Bao Lab](https://baulab.info/)'s [April 2026 puzzle: Max of List](https://puzzles.baulab.info/april-2026.html).

---
## Table of Contents

- Setup code: Loads helper modules.
- Sidebar: We provide counterexamples showing that Puzzle 2 is not 100% accurate.
- Sampling code: Samples activations from each model.

### Puzzle 1
We introduce three experimental techniques that take advantage of the model's linearity (i.e. absence of layer norms), using them to solve Puzzle 1.

- We show that a single attention head is sufficiently powerful to immediately find the maximum digit.

1. (1-1) Linearity allows us to independently analyze initial-layer attention scores: `S[E + P] = S[E] + S[P]` (E=embeds, P=positional embeds).
    - Using this, we show how head 3 uses E and P to take the max (and heads 0&2 attend to digits 7-9).
2. (1-2) We use logistic regression on head residual contributions (after partitioning W_O, for head hi: `W_V[hi] @ W_O[hi]`). By creating a custom per-head unembedding, we determine whether each head's output "knows" the true class.
    - This supports (1-1) by showing head 3 output perfectly "knows" the max digit, while heads 0&2 often know it.
3. (1-3) We compute attention head logit attributions (by linearity: `W_V[hi] @ W_O[hi] @ W_U.T`).
    - We show that each head either (A) outputs info about the true class`*`, or (B) contributes a surprisingly uniform per-class bias.
    - This supports results from (1-1) and (1-2). It clearly shows that heads 0&2 outputs reinforce head 3 in knowing and promoting only true classes 7-9, which account for >67% of the maxes in all possible sequences.

### Puzzle 2

- (2-1) We first broadly visualize the below mechanism using a grid of logistic regression results.

Tens digit:
1. (2-2) Layer 0 Head 1 (L0H1) identifies the max tens digit as in Puzzle 1, copying it to the ANS residual row.
2. (2-5) Before layer 1, the max tens digit is already known and on the ANS-row residual stream.
    - Using logit attribution, we show that layer 1 heads (A) reinforce this true class (especially heads 1&2) and (B) otherwise provide per-class constants to properly match the unembeddings.

Ones digit:
1. (2-3) L0H0/2 (heads 0 and 2): Attend to the ANS row, which we know from above contains the tens digit. This info is copied to the ANS+1 residual row.
2. (2-4) L0H3: Copies each tens digit into the corresponding ones-digit residual activation rows.
3. (2-6) L1H0/3: Recognize the max ones digit by applying: (A) a score boost when (2-4) matches with (2-3), (B) a score gradient to the ones digits themselves, and (C) a mask on the ones digits via the positional embeds.

### Appendices
1. (3-1) Ablation experiment results. Interactively supports earlier results and allows viewing of model biases.
2. (3-2) Attention pattern explorer.

`*` We refer to "output class" as a possible output token, and "true class" as the *correct* token. Further, when we state "copies/contains the tens digit", we mean that some information about the tens digit is copied/contained but using a possibly-different representation, e.g. transformed by W_VO.