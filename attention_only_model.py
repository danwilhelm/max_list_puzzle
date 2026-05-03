import torch
import torch.nn.functional as F
import numpy as np

# Numpy implementation of attention-only model
class AttentionOnlyModel:
    @staticmethod
    def softmax(x):
        exp_x = np.exp(x - x.max(axis=-1, keepdims=True)[0])  # avoid overflows via translation invariance
        return exp_x / (np.sum(exp_x, axis=-1, keepdims=True))
        #return F.softmax(torch.tensor(x), dim=-1).cpu().numpy()

    def __init__(self, cfg, state_dict):
        self.d_vocab = cfg['model']['vocab_size']
        self.d_model = cfg['model']['d_model']
        self.n_heads = cfg['model']['n_heads']
        self.n_layers = cfg['model']['n_layers']
        self.n_ctx = cfg['model']['max_seq_len']
        self.vocab_info = cfg['vocab']
        self.training_info = cfg['training']

        self.num_range = cfg['training']['num_range']
        self.list_len = cfg['training']['list_len']
        self.n_digits = 10

        self.d_head = self.d_model // self.n_heads
        self.ignore = -np.inf

        self.W_E = state_dict['tok_embed.weight'].cpu().numpy()
        self.W_P = state_dict['pos_embed.weight'].cpu().numpy()
        self.W_U = state_dict['unembed.weight'].cpu().numpy()

        self.W_Q = np.empty((self.n_layers, self.n_heads, self.d_model, self.d_head), dtype=np.float32)
        self.W_K = np.empty_like(self.W_Q)
        self.W_V = np.empty_like(self.W_Q)
        self.W_O_full = np.empty((self.n_layers, self.d_model, self.d_model), dtype=np.float32)

        for layer in range(self.n_layers):
            self.W_O_full[layer] = state_dict[f'layers.{layer}.W_O.weight'].cpu().numpy().T
            for head in range(self.n_heads):
                self.W_Q[layer, head] = state_dict[f'layers.{layer}.heads.{head}.W_Q.weight'].T.cpu().numpy()
                self.W_K[layer, head] = state_dict[f'layers.{layer}.heads.{head}.W_K.weight'].T.cpu().numpy()
                self.W_V[layer, head] = state_dict[f'layers.{layer}.heads.{head}.W_V.weight'].T.cpu().numpy()

        # For computing residual stream contribution per head
        self.W_O = self.W_O_full.reshape((self.n_layers, self.n_heads, self.d_head, self.d_model))
        self.W_QK = self.W_Q @ self.W_K.transpose([0,1,3,2])
        self.W_VO = self.W_V @ self.W_O


    def alloc_outputs(self, toks_in):
        n_examples, n_toks = toks_in.shape
        n_layers, n_heads, d_head = self.n_layers, self.n_heads, self.d_head

        self.attn_in = np.empty((n_layers, n_examples, n_toks, self.d_model))
        self.q = np.empty((n_layers, n_examples, n_heads, n_toks, d_head))
        self.k = np.empty((n_layers, n_examples, n_heads, n_toks, d_head))
        self.v = np.empty((n_layers, n_examples, n_heads, n_toks, d_head))

        self.scores = np.empty((n_layers, n_examples, n_heads, n_toks, n_toks))
        self.pattern = np.empty_like(self.scores)
        self.z = np.empty((n_layers, n_examples, n_heads, n_toks,d_head))
        self.attn_result = np.empty((n_layers, n_examples, n_heads, n_toks, self.d_model))
        self.attn_logits = np.empty((n_layers, n_examples, n_heads, n_toks, self.d_vocab))
        self.attn_out = np.empty_like(self.attn_in)


    def run(self, toks_in, use_pos_embeds=True):
        layer = 0
        if type(toks_in) is list:
            toks_in = np.array(toks_in)
        elif type(toks_in) is torch.Tensor:
            toks_in = toks_in.detach().cpu().numpy()

        if toks_in.ndim == 1:
            toks_in = toks_in[None]

        if toks_in.shape[-1] == self.d_model and toks_in.dtype == np.float32:
            # Input: Residual stream
            self.toks_in = None
            self.n_toks = toks_in.shape[1]    # (n_examples, n_toks,d_model)  <- we omit n_examples from most below
            self.embeds_in, self.pos_embeds_in = None, None
            self.stream_in = toks_in
        else:
            # Input: Token indexes
            self.toks_in = toks_in.copy()
            self.n_toks = self.toks_in.shape[-1]
            self.embeds_in = self.W_E[self.toks_in]
            self.pos_embeds_in = self.W_P[:self.n_toks] if use_pos_embeds else np.zeros_like(self.embeds_in)
            self.stream_in = self.embeds_in + self.pos_embeds_in  # (n_examples, n_toks,d_model)  <- we omit n_examples from most below

        self.mask = np.tril(np.ones((self.n_toks, self.n_toks)))

        self.alloc_outputs(toks_in)  # pre-allocate memory for attention (cleans up the for loop implementation)

        # ATTENTION
        block_out = self.stream_in                  # (n_toks,d_model)
        for layer in range(self.n_layers):
            self.attn_in[layer] = block_out

            # QK compute the attn weights W. Then, VO transform the weighted inputs WX.
            self.q[layer] = block_out[:,None] @ self.W_Q[layer]  # (n_toks,d_model) @ (n_heads, d_model,d_head)
            self.k[layer] = block_out[:,None] @ self.W_K[layer]  #   => (n_heads, n_toks,d_head)
            self.v[layer] = block_out[:,None] @ self.W_V[layer]  #

            # (n_heads, n_toks,d_model) @ (d_model,n_toks) => (n_heads, n_toks,n_toks)
            self.scores[layer] = self.q[layer] @ self.k[layer].transpose([0,1,3,2]) / np.sqrt(self.d_head)
            masked_scores = np.where(self.mask[:self.n_toks, :self.n_toks],        # causal attn -- per row, mask out prior tokens
                                     self.scores[layer],
                                     self.ignore)        # a very small value (or -inf) so that softmax gives it a 0
            self.pattern[layer] = self.softmax(masked_scores)  # (n_heads, n_toks,n_toks)
            self.z[layer] = self.pattern[layer] @ self.v[layer]     # (n_heads, n_toks,n_toks) @ (n_heads, n_toks,d_head) => (n_heads, n_toks,d_head)

            # (n_heads, n_toks,d_head) @ (n_heads, d_head,d_model) => (n_heads, n_toks,d_model)
            self.attn_result[layer] = self.z[layer] @ self.W_O[layer]      # attn_result is the contribution of each head to the residual stream
            self.attn_logits[layer] = self.attn_result[layer] @ self.W_U.T  # direct logit attribution (if no additional layers/layernorms/etc)

            self.attn_out[layer] = np.sum(self.attn_result[layer], axis=1)   # (n_examples, *n_heads*, n_toks,d_head) => (n_examples, n_toks,d_head)
            block_out = self.attn_in[layer] + self.attn_out[layer]

        # UNEMBED
        self.stream_out = block_out                       # (n_toks,d_model) - add skip connection
        self.logits = self.stream_out @ self.W_U.T        # (d_toks,d_model) @ (d_model,d_vocab_out) + (d_model,) => (n_toks,d_vocab_out)
        self.probs = self.softmax(self.logits)            # (n_toks,d_vocab_out)
        self.labels = np.argmax(self.probs, axis=-1)      # (n_toks,)

        return self.logits

    def run_ablate_head(self, toks_in, layer_ix, head_ix, use_pos_embeds=True):
        WO = self.W_O.copy()
        try:
            self.W_O[layer_ix, head_ix] = 0.0
            logits = self.run(toks_in, use_pos_embeds=use_pos_embeds)
        finally:  # ensures WO is replaced, even upon error (e.g. keyboard interrupt)
            self.W_O = WO

        return logits

    def quick_attn(self, layer, head, rows, cols, softmax=True):
        # Same as (XQ) @ (XK).T / sqrt(d_head)
        scores = np.array([[np.sum(np.tensordot(row, col, axes=0) * self.W_QK[layer, head]) / np.sqrt(self.d_head) \
                            for row in rows] for col in cols]).T
        # causal attn -- per row, mask out prior tokens
        if softmax:
            scores = np.where(np.tril(np.ones((len(scores), len(scores)))),
                              scores,
                              self.ignore)
            scores = self.softmax(scores)
        return scores

    def verify_raw_model(self, toks, raw_model, atol=1e-3):
        # Verify models equivalent. Note there is some tolerance due to low-level algorithms and different matrix factorings.
        device = raw_model.tok_embed.weight.device

        results = []
        raw_logits_1, _ = raw_model(torch.tensor(toks, device=device))
        logits_1 = self.run(toks)
        same_logits = np.all(np.isclose(raw_logits_1.detach().cpu().numpy(), logits_1, atol=atol))
        next_tok = np.argmax(logits_1[:,-1], axis=-1)
        results.append([torch.argmax(raw_logits_1[:,-1], dim=-1), next_tok])

        if same_logits and self.num_range > 10:
            toks = np.hstack([toks, next_tok[:,None]])
            raw_logits_2, _ = raw_model(torch.tensor(toks, device=device))
            logits_2 = self.run(toks)
            same_logits = np.all(np.isclose(raw_logits_2.detach().cpu().numpy(), logits_2, atol=atol))
            results.append([torch.argmax(raw_logits_2[:,-1], dim=-1), np.argmax(logits_2[:,-1], axis=-1)])

        return results, same_logits
