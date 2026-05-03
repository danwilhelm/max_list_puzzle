import json, importlib
from pathlib import Path

import torch
import numpy as np

import plotly.graph_objects as go
from plotly.subplots import make_subplots

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from sklearn.linear_model import LogisticRegression
from huggingface_hub import hf_hub_download

from attention_only_model import AttentionOnlyModel

examples_1 = [
    [3, 7, 2, 5, 1],
    [0, 0, 0, 0, 0],
    [9, 1, 8, 2, 7],
    [1, 2, 3, 4, 5],
]

examples_2 = [
    [42, 17, 85, 3, 61],
    [99, 0, 50, 25, 75],
    [87, 86, 85, 84, 83],
    [9, 19, 29, 39, 49],
]

counterexample_seqs = [
    [ 0,  0,  4, 57, 63],
    [52, 49, 49, 28,  0],
    [81,  2, 10,  0,  7],
    [80, 12,  0,  0,  7],
    [45, 44, 44, 31, 23],
    [13, 15, 14, 14, 14],
]


def where(cond):
    return np.nonzero(cond)[0]

def wherein(elements, test_elements):
    """Returns the index of each `elements` item in `test_elements`"""
    return np.array([where(test_elements == num)[0] for num in elements])


def tokenize_1(nums: list[int]) -> list[int]:
    """Tokenize a list of numbers for Model 1.
    Example: [3, 7, 2] -> [BOS, 3, SEP, 7, SEP, 2, ANS]"""
    BOS, SEP, ANS, EOS = 10, 11, 12, 13

    tokens = [BOS]
    for i, n in enumerate(nums):
        tokens.append(n)
        if i < len(nums) - 1:
            tokens.append(SEP)
    tokens.append(ANS)
    return tokens

def tokenize_2(nums: list[int]) -> list[int]:
    """Tokenize a list of numbers for Model 2.
    Example: [42, 7, 85] -> [BOS, 4, 2, SEP, 0, 7, SEP, 8, 5, ANS]"""
    BOS, SEP, ANS, EOS = 10, 11, 12, 13
    tokens = [BOS]
    for i, n in enumerate(nums):
        tokens.append(n // 10)
        tokens.append(n % 10)
        if i < len(nums) - 1:
            tokens.append(SEP)
    tokens.append(ANS)
    return tokens


def load_models(device='cpu'):
    # Download model definition and both sets of weights from HuggingFace
    model_py_path = hf_hub_download("andyrdt/04_2026_puzzle_1a", "model.py")
    spec = importlib.util.spec_from_file_location("model", model_py_path)
    model_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_module)
    AttentionOnlyTransformer = model_module.AttentionOnlyTransformer

    # Pre-download both models so later cells don't need network access
    config_1_path = hf_hub_download("andyrdt/04_2026_puzzle_1a", "config.json")
    weights_1_path = hf_hub_download("andyrdt/04_2026_puzzle_1a", "model.pt")
    config_2_path = hf_hub_download("andyrdt/04_2026_puzzle_1b", "config.json")
    weights_2_path = hf_hub_download("andyrdt/04_2026_puzzle_1b", "model.pt")

    print("Downloaded model definition + weights for both models.")

    # Load Model 1
    config_1 = json.loads(Path(config_1_path).read_text())
    state_dict_1 = torch.load(weights_1_path, map_location=device, weights_only=True)
    raw_model_1 = AttentionOnlyTransformer.from_config(config_1["model"])
    raw_model_1.load_state_dict(state_dict_1)
    raw_model_1.eval().to(device)

    config_2 = json.loads(Path(config_2_path).read_text())
    state_dict_2 = torch.load(weights_2_path, map_location=device, weights_only=True)
    raw_model_2 = AttentionOnlyTransformer.from_config(config_2["model"])
    raw_model_2.load_state_dict(state_dict_2)
    raw_model_2.eval().to(device)

    model1 = AttentionOnlyModel(config_1, state_dict_1)
    model2 = AttentionOnlyModel(config_2, state_dict_2)

    return raw_model_1, raw_model_2, model1, model2


##################################################
# Analysis/Graphics used in the notebook
##################################################

def imshow(matrix, center=False, facet_col=None, sharey=False,
           colorbar=False, colorbar_scale=0.8,
           xticks=None, yticks=None, xlabel=None, ylabel=None,
           vmin=None, vmax=None,
           title=None, cmap='seismic', figsize=None, axs=None):
    """Note: Only first-dimensional faceting works (facet_col=0) :D"""
    # seismic is best for centering (white=0)
    cols = 1 if facet_col is None else matrix.shape[0]
    if axs is None: fig,axs = plt.subplots(1,cols, figsize=figsize, sharey=sharey)
    else: fig = axs.get_figure()
    vmax = vmax or (max(abs(matrix.max()), abs(matrix.min())) if center else matrix.max())
    vmin = vmin or (-vmax if center else matrix.min())

    for i in range(cols):
        ax = axs if cols == 1 else axs[i]
        ax.imshow(matrix if cols == 1 else matrix[i], cmap=cmap, vmax=vmax, vmin=vmin)
        if xlabel is not None: ax.set_xlabel(xlabel)
        if ylabel is not None: ax.set_ylabel(ylabel)
        if xticks is not None: ax.set_xticks(np.arange(len(xticks)), xticks)
        if yticks is not None: ax.set_yticks(np.arange(len(yticks)), yticks)
        if title is not None: ax.set_title(title.strip() + (f' {i}' if cols > 1 else ''))

    if colorbar is not None:
        fig.colorbar(ScalarMappable(norm=Normalize(vmin, vmax), cmap='seismic'), ax=axs, shrink=colorbar_scale)
    return fig,axs


def attn_logit_means(model, y_true, tens_digit=False):
    """For all examples of a given y, compute mean/std of each class logit."""
    pos = -2 if tens_digit else -1
    means = np.empty((model.n_layers, model.n_digits,model.n_digits, model.n_heads))
    stds = np.empty_like(means)
    for layer in range(model.n_layers):
        for y in range(model.n_digits):
            example_ixs = where(y_true // 10 == y) if tens_digit else where(y_true % 10 == y)
            if len(example_ixs) > 0:
                for out_cls in range(model.n_digits):
                    means[layer,y,out_cls] = model.attn_logits[layer,example_ixs, :,pos, out_cls].mean(axis=0)
                    stds[layer,y,out_cls] = model.attn_logits[layer,example_ixs, :,pos, out_cls].std(axis=0)
            else:
                means[layer,y,:] = 0.
                stds[layer,y,:] = 0.
    return means, stds


def show_attn_gradient(model, layer, title='', sub_pos=None, row=None, cols=None):
    sub_pos = sub_pos or model.ANS
    special_pos = [model.BOS_POS, model.SEP_POS, model.ANS_POS, model.LAST_POS]

    if row is None: row = model.W_E[model.ANS] + model.W_P[model.ANS_POS]   # last row before predicting the max digit
    if cols is None: cols = np.vstack([model.W_E + model.W_P[[model.BOS_POS+1]*model.n_digits + special_pos], model.W_P])
    all_labels = model.embed_labels[:len(cols)] + model.seq_labels

    # Same as (XQ) @ (XK).T / sqrt(d_head)
    scores = np.array([[np.sum(np.tensordot(row, col, axes=0) * model.W_QK[layer,head]) / np.sqrt(model.d_head) \
                        for col in cols] for head in range(model.n_heads)])
    scores -= scores[:,sub_pos][:,None]  # subtract a per-row constant (softmax is translation-invariant)

    fig, axs = imshow(scores, center=True, xlabel='embed/positional token', ylabel='head', figsize=(16,6),
                      xticks=all_labels[:len(cols)], yticks=np.arange(model.n_heads), colorbar=True, colorbar_scale=0.5,
                      title=title)
    axs.axvline(len(model.embed_labels[:-1])-0.5, c='k')


def show_logit_contribs(model, means, stds, layer=0, title='', xaxis_ytrue=False):
    fig = make_subplots(rows=2,cols=2, subplot_titles=[f'Head {i}' for i in range(model.n_heads)],
                        horizontal_spacing=0.07, vertical_spacing=0.2)
    for head in range(model.n_heads):
        row, col = head//2+1, head%2+1
        for y in range(model.n_digits):
            curr_means = means[layer, :,y,head] if xaxis_ytrue else means[layer, y,:,head]
            curr_stds = stds[layer, :,y,head] if xaxis_ytrue else stds[layer, y,:,head]
            scale = ['rgb(20,200,0)' if i == y and y>0 else f'rgb({y*255/model.n_digits},50,50)' for i in range(model.n_digits)]
            trace = go.Bar(x=np.arange(0,model.n_digits), y=curr_means, marker_color=scale,
                           error_y=dict(type='data', array=curr_stds, color='red'), name=f'{y}', showlegend=head == 0)
            fig.add_trace(trace, row=row, col=col)
            fig.update_xaxes(title_text='TRUE class' if xaxis_ytrue else 'OUTPUT class', row=row, col=col, tickmode='linear')
            fig.update_yaxes(title_text='head logit contribution', row=row, col=col)
            fig.add_hline(0, row=row, col=col)

    fig.update_layout(
        title=title,
        legend=dict(title=dict(text='OUTPUT class' if xaxis_ytrue else 'TRUE class')),
        height=600
    )
    fig.show()
    return fig


def unembed_accuracy(model, X, y_true, n_samples=1000):
    y_pred = np.argmax(X[n_samples:n_samples*2] @ model.W_U.T, axis=-1)
    return np.mean(y_pred == y_true[n_samples:n_samples*2])


def logistic_accuracy(X, y, n_samples=1000, max_iter=10000):
    """Assumes X and y contain double `n_samples` (for validation set) """
    lr = LogisticRegression(max_iter=max_iter, solver='newton-cg')  # using newton since no max_iter satisfies lbfgs
    lr.fit(X[:n_samples], y[:n_samples])            # train on training set
    accuracy = lr.score(X[n_samples:n_samples*2],
                        y[n_samples:n_samples*2])   # test on validation set
    return lr, accuracy


def logistic_head_grid(model, y_true, use_unembed=False,
                       use_attn_out=False, n_samples=1000):
    """Returns (n_layers, tens pos=0/ones pos=1, n_heads, tens/ones)"""
    probas = np.empty((model.n_layers*model.n_heads, 2, 2))
    head_labels = []
    for layer in range(model.n_layers):
        for head in range(model.n_heads):
            head_labels.append(f'layer {layer}\nhead {head}')
            for pos in [-2, -1]:
                if use_attn_out:
                    X = model.attn_out[layer,:,pos]
                else:
                    X = model.attn_result[layer,:,head,pos]
                if use_unembed: acc = unembed_accuracy(model, X, y_true // 10, n_samples=n_samples)
                else: _,acc = logistic_accuracy(X, y_true // 10, n_samples=n_samples)
                probas[layer*model.n_heads+head,0,pos+2] = acc

                if use_unembed: acc = unembed_accuracy(model, X, y_true % 10, n_samples=n_samples)
                else: _,acc = logistic_accuracy(X, y_true % 10, n_samples=n_samples)
                probas[layer*model.n_heads+head,1,pos+2] = acc
    return probas, head_labels
