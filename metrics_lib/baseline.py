"""
Baseline single-text metrics: lexical Diversity, OPT-2.7B Perplexity/Coherence and the combined Q*Text score.

- Definitions from Garces Arias et al., 2025: Towards Better Open-Ended Text
Generation: A Multicriteria Evaluation Framework (GEM^2 2025)
- OPT-2.7B is used as a fixed external scorer for both Perplexity and Coherence,
matching the paper's own Coherence scorer (the paper itself uses two different
scorers per metric -> here one shared scorer covers both for simplicity)
- Q*Text reuses the paper's fitted parameter

    from metrics_lib.baseline import diversity_score, compute_perplexity, compute_coherence, compute_qstar
    div = diversity_score(text)
    ppl = compute_perplexity(text)
    coh = compute_coherence(prompt, text)
    qstar = compute_qstar(perplexity_array, coherence_array, diversity_array)  # dataset-relative, see below
"""
import re
from functools import lru_cache

import numpy as np

DEFAULT_MODEL_NAME = 'facebook/opt-2.7b'
DEFAULT_MAX_TOKENS = 256

# Fitted hyperparameters from Garces Arias et al. (2025)
# Order in each tuple: (Perplexity, Coherence, Diversity)
Q_STAR_PARAMS = {
    'w':     (0.586, 0.834, 3.853),   # weight: how much each metric contributes to the final score
    'mu':    (0.458, 0.000, 0.854),   # target: the "ideal" normalized value for each metric
    'alpha': (2.579, 1.496, 7.370),   # penalty strength: how sharply the score drops away from mu
}

# Simple tokenizer for diversity score (no punctuation, lowercase)
def word_tokenize_simple(text):
    return re.findall(r"\w+", text.lower())



def diversity_score(text, n_range=(2, 3, 4)):
    """DIV(x) = product over n in n_range of (unique n-grams / total n-grams). NaN if text too short."""
    # Tokenize text
    tokens = word_tokenize_simple(text)

    score = 1.0
    for n in n_range:

        # At least n tokens are needed to form a single n-gram; otherwise return NaN
        if len(tokens) < n:
            return np.nan

        # Compute n-grams and their uniqueness
        ngrams = list(zip(*(tokens[i:] for i in range(n))))
        unique = len(set(ngrams))

        # Update score with the ratio of unique n-grams to total n-grams
        score *= unique / len(ngrams)
    return score


# Shared scorer for Perplexity and Coherence
# Cached to avoid reloading the model/tokenizer
@lru_cache(maxsize=None)
def _load_scorer(model_name=DEFAULT_MODEL_NAME, device=None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Determine device: MPS (Apple Silicon) if available, else CPU
    if device is None:
        device = 'mps' if torch.backends.mps.is_available() else 'cpu'

    # float16 on GPU/MPS to keep a multi-billion-parameter model's memory footprint manageable
    # CPU falls back to float32 since fp16 matmul kernels are unsupported/slow there
    dtype = torch.float16 if device != 'cpu' else torch.float32

    # Load tokenizer and model, move model to device and set to eval mode
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype).to(device)
    model.eval()

    return tokenizer, model, device


def compute_perplexity(text, model_name=DEFAULT_MODEL_NAME, max_tokens=DEFAULT_MAX_TOKENS):
    """Perplexity of `text` alone (no prompt context) under the scorer model. NaN if too short."""

    # Load the scorer (tokenizer and model) using the cached function
    import torch
    tokenizer, model, device = _load_scorer(model_name)

    # Tokenize the input text, ensuring it fits within the max token limit and move to the appropriate device
    input_ids = tokenizer(text, return_tensors='pt', truncation=True, max_length=max_tokens)['input_ids'].to(device)

    # If the tokenized input is too short to compute perplexity, return NaN
    if input_ids.shape[1] < 2:
        return np.nan

    # Compute the loss (negative log-likelihood) of the input text under the model and exponentiate to get perplexity
    with torch.no_grad():
        loss = model(input_ids, labels=input_ids).loss

    return float(torch.exp(loss).cpu())


def compute_coherence(prompt, text, model_name=DEFAULT_MODEL_NAME, max_tokens=DEFAULT_MAX_TOKENS):
    """Mean log-probability of `text`'s tokens conditioned on `prompt`, under the scorer model."""

    import torch
    tokenizer, model, device = _load_scorer(model_name)

    # Tokenize the prompt and continuation, ensuring they fit within the max token limit and move to appropriate device.
    # add_special_tokens=False on the continuation: some tokenizers (e.g. OPT's, which prepends
    # </s> as BOS) would otherwise insert a spurious special token right at the prompt/continuation
    # boundary once concatenated below.
    prompt_ids = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=max_tokens)['input_ids']
    cont_ids = tokenizer(text, return_tensors='pt', truncation=True, max_length=max_tokens,
                          add_special_tokens=False)['input_ids']

    # If the continuation is too short to compute coherence, return NaN
    if cont_ids.shape[1] < 1:
        return np.nan

    # Concatenate prompt and continuation token IDs
    input_ids = torch.cat([prompt_ids, cont_ids], dim=1)

    #  Determine the length of the prompt
    prompt_len = prompt_ids.shape[1]

    # If the combined input exceeds the model's context limit, truncate the oldest tokens from the beginning.
    # Context-window field differs by architecture (GPT-2: n_positions, OPT/most others: max_position_embeddings)
    ctx_limit = getattr(model.config, 'n_positions', None) or model.config.max_position_embeddings

    if input_ids.shape[1] > ctx_limit:
        # Truncate the oldest tokens from the beginning to fit within the context limit
        overflow = input_ids.shape[1] - ctx_limit
        input_ids = input_ids[:, overflow:]
        prompt_len = max(prompt_len - overflow, 0)

    # Move the input IDs to the appropriate device
    input_ids = input_ids.to(device)

    # Compute the log-probabilities of the continuation tokens conditioned on the prompt
    with torch.no_grad():
        logits = model(input_ids).logits

    log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
    targets = input_ids[:, 1:]
    token_log_probs = log_probs.gather(2, targets.unsqueeze(-1)).squeeze(-1)

    # Compute the mean log-probability of the continuation tokens, starting from the end of the prompt
    cont_start = max(prompt_len - 1, 0)
    cont_log_probs = token_log_probs[:, cont_start:]

    # Return NaN if the continuation has no tokens
    if cont_log_probs.shape[1] == 0:
        return np.nan
    
    return float(cont_log_probs.mean().cpu())


def gaussian_penalty(x, mu, alpha):
    """P_i(x) = exp(-alpha * (x - mu)^2): 1.0 at x == mu, decaying away from it."""
    return np.exp(-alpha * (np.asarray(x, dtype=float) - mu) ** 2)


def compute_qstar(perplexity, coherence, diversity, params=Q_STAR_PARAMS):
    """
    Q*Text = weighted sum of Gaussian-penalized, dataset-relative min-max-normalized metrics, x100.

    perplexity/coherence/diversity must be array-likes covering the full set
    of texts you want compared against each other (they're min-max
    normalized against each other's min/max) -> not single scalars.
    """
    perplexity = np.asarray(perplexity, dtype=float)
    coherence = np.asarray(coherence, dtype=float)
    diversity = np.asarray(diversity, dtype=float)

    # Min-max normalize each metric across the dataset, flipping perplexity's direction since lower is better
    p_min, p_max = perplexity.min(), perplexity.max()
    c_min, c_max = coherence.min(), coherence.max()
    d_min, d_max = diversity.min(), diversity.max()

    # Compute normalized metrics (m1, m2, m3) for perplexity, coherence, and diversity
    m1 = (p_max - perplexity) / (p_max - p_min)   # Perplexity: raw LOWER is better -> flip direction
    m2 = (coherence - c_min) / (c_max - c_min)    # Coherence: raw higher is better
    m3 = (diversity - d_min) / (d_max - d_min)    # Diversity: raw higher is better

    # Apply Gaussian penalties to each normalized metric using the fitted parameters (mu, alpha) from Q_STAR_PARAMS
    w1, w2, w3 = params['w']
    mu1, mu2, mu3 = params['mu']
    a1, a2, a3 = params['alpha']

    p1 = gaussian_penalty(m1, mu1, a1)
    p2 = gaussian_penalty(m2, mu2, a2)
    p3 = gaussian_penalty(m3, mu3, a3)

    # Compute the weighted sum of the penalized metrics and scale by 100 to get the final Q*Text score
    numerator = w1 * m1 * p1 + w2 * m2 * p2 + w3 * m3 * p3
    return 100 * numerator / (w1 + w2 + w3)
