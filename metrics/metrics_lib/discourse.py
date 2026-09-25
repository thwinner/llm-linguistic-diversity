"""
Discourse diversity (D_disc): entity-grid-based diversity metric

Two variants of the entity-grid model based on Barzilay & Lapata (2008):

- discourse_diversity(): role-based variant -> per-entity grammatical roles
  (Subject / Object / Other / Absent), salience-weighted, weighted bigram +
  trigram role-transition profiles
- discourse_diversity_simple(): simplified variant -> binary presence/absence
  per entity (no roles, no salience weighting), single set of four transition types.

Both reduce to a mean pairwise Jensen-Shannon divergence across a group of texts.

    from metrics_lib.discourse import discourse_diversity, discourse_diversity_simple
    d_disc = discourse_diversity(list_of_texts_for_one_prompt)
    d_disc_simple = discourse_diversity_simple(list_of_texts_for_one_prompt)

- Entities follow Barzilay & Lapata's no-coreference setting: the head of every
noun chunk (POS NOUN or PROPN), clustered by its normalized lemma. 
- Both variants share this single extraction (extract_entities)
- Role-based variant additionally reads off the head's grammatical role
- A text with no usable entity profile yields no profile at all (None) and a group with 
fewer than two usable profiles yields np.nan
"""
from collections import Counter
from itertools import combinations
import numpy as np
from scipy.stats import entropy
from ._common import get_spacy_nlp


# Entity-grid roles (Barzilay & Lapata, 2008)
# S = subject, O = object, X = other, - = absent
ROLE_S, ROLE_O, ROLE_X, ROLE_ABSENT = "S", "O", "X", "-"
ROLES = [ROLE_S, ROLE_O, ROLE_X, ROLE_ABSENT]

# All possible bigram and trigram transitions between roles (for profile vectors)
BIGRAM_TRANSITIONS = [(r1, r2) for r1 in ROLES for r2 in ROLES]
TRIGRAM_TRANSITIONS = [(r1, r2, r3) for r1 in ROLES for r2 in ROLES for r3 in ROLES]
N_BIGRAM = len(BIGRAM_TRANSITIONS)    # 16
N_TRIGRAM = len(TRIGRAM_TRANSITIONS)  # 64

# Entity must appear in >= this many sentences to count
# (detailed analysis of the entity filter in metrics/01_discourse_diversity/disc_diversity.ipynb)
MIN_ENTITY_FREQ = 2

# Salience weighting scheme for role transitions of an entity seen in f sentences.
# Robustness check in metrics/01_discourse_diversity/disc_diversity.ipynb.
#   "none" -> w_e = 1
#   "log"  -> w_e = log(1 + f_e)   (default)
#   "freq" -> w_e = f_e
WEIGHT_SCHEME = "log"

# Grammatical dependency labels for subjects and objects (spaCy)
_SUBJ_DEPS = {"nsubj", "nsubjpass", "csubj", "csubjpass"}
_OBJ_DEPS = {"dobj", "iobj", "pobj", "attr", "oprd", "dative"}

# Head POS tags that count as a discourse entity
_ENTITY_HEAD_POS = {"NOUN", "PROPN"}

# All possible presence/absence transitions (for the simplified variant's profile vector)
BINARY_TRANSITIONS = [(a, b) for a in (0, 1) for b in (0, 1)]
N_BINARY = len(BINARY_TRANSITIONS)  # == 4


def _entity_weight(freq, scheme=None):
    """Salience weight for an entity appearing in `freq` sentences (see WEIGHT_SCHEME)."""

    scheme = scheme or WEIGHT_SCHEME
    if scheme == "none":
        return 1.0
    if scheme == "freq":
        return float(freq)
    if scheme == "log":
        return float(np.log1p(freq))
    
    raise ValueError(f"unknown weight scheme: {scheme!r}")


def _get_role_for_token(token):
    """Grammatical role of a token, walking up the dependency chain to the
    governing subject/object. Returns ROLE_S, ROLE_O, or ROLE_X (other)."""

    node = token
    while True:
        if node.dep_ in _SUBJ_DEPS:
            return ROLE_S
        if node.dep_ in _OBJ_DEPS:
            return ROLE_O
        if node.head == node:
            return ROLE_X
        node = node.head


def extract_entities(sent_doc):
    """Discourse entities for one spaCy-parsed sentence: 
    Head of every noun chunk, kept when its POS is NOUN or
    PROPN, clustered by the head's normalized lemma. Each entity is tagged with
    its grammatical role (priority S > O > X).

    Returns {entity_id: role}.
    """
    entity_roles = {}
    role_priority = {ROLE_S: 3, ROLE_O: 2, ROLE_X: 1}

    # Walk through all noun chunks in the sentence and extract their head tokens
    for chunk in sent_doc.noun_chunks:
        head = chunk.root

        # Only noun / proper-noun heads count as entities (drops pronoun-headed chunks)
        if head.pos_ not in _ENTITY_HEAD_POS:
            continue

        # Entity id = normalized lemma of the head noun
        name = head.lemma_.lower().strip()

        # Ignore very short lemmas (e.g., single-character pronouns, punctuation, etc.)
        if len(name) < 2:
            continue

        # Role from the head token, walking up the dependency chain
        role = _get_role_for_token(head)

        # Keep the highest-priority role for each entity (S > O > X)
        if name not in entity_roles or role_priority[role] > role_priority[entity_roles[name]]:
            entity_roles[name] = role

    return entity_roles


def extract_entities_simple(sent_doc):
    """Set of discourse entity ids in one sentence (roles ignored) -> same
    extraction as extract_entities(), used by the simplified binary grid."""
    return set(extract_entities(sent_doc))


def build_entity_grid(text, nlp=None):
    """Entity grid for one text

    Returns (grid, sents, entity_freq):
      grid: {entity_name: [role_sent1, role_sent2, ...]}, role in {S, O, X, -}
      sents: list of sentence spans from the single parse (alpha-only, no
        minimum-length threshold)
      entity_freq: {entity_name: number of sentences it appears in}
    """

    # Parse the text once; entity extraction runs directly on the resulting
    # sentence spans (no second parse over re-joined sentence strings)
    nlp = nlp if nlp is not None else get_spacy_nlp()
    doc = nlp(text)
    sents = [s for s in doc.sents if any(ch.isalpha() for ch in s.text)]

    # If there are fewer than 2 sentences, return empty structures since we can't compute transitions
    if len(sents) < 2:
        return {}, sents, {}

    sent_entity_roles = [extract_entities(s) for s in sents]

    # Collect all unique entities across sentences to build the grid and count their frequencies
    all_entities = set()
    for roles in sent_entity_roles:
        all_entities.update(roles.keys())

    if not all_entities:
        return {}, sents, {}

    # Initialize the grid and entity frequency dictionary
    grid, entity_freq = {}, {}

    # For each entity, create a list of its roles across sentences (or '-' if absent) and count how many sentences it appears in
    for entity in all_entities:
        roles_seq, freq = [], 0
        for sent_roles in sent_entity_roles:
            if entity in sent_roles:
                roles_seq.append(sent_roles[entity])
                freq += 1
            else:
                roles_seq.append(ROLE_ABSENT)
        grid[entity] = roles_seq
        entity_freq[entity] = freq

    return grid, sents, entity_freq


def compute_transition_profile(text, use_trigrams=True, nlp=None, weight_scheme=None):
    """Weighted bigram/trigram role-transition profile for one text.

    - Salience weighting w_e (see WEIGHT_SCHEME / weight_scheme; default log(1 + freq(e)))
    - Entities below MIN_ENTITY_FREQ are dropped

    Returns (bigram_profile, trigram_profile, grid, sents, entity_freq).
      bigram_profile is None when the text has no usable entity profile
        (< 2 sentences, no entities, or none passing the frequency filter).
      trigram_profile is None when trigrams are disabled or no surviving entity
        spans >= 3 sentences.
    """

    # Use spaCy to build the entity grid and extract sentences and entity frequencies
    grid, sents, entity_freq = build_entity_grid(text, nlp=nlp)

    # No entities / too few sentences -> no usable profile (NA, not a uniform fallback)
    if not grid or len(sents) < 2:
        return None, None, grid, sents, entity_freq

    # Compute weighted bigram and trigram counts for the entity roles across sentences, applying salience weighting based on entity frequency
    bigram_counts, trigram_counts = Counter(), Counter()
    for entity, roles_seq in grid.items():
        if entity_freq[entity] < MIN_ENTITY_FREQ:
            continue
        weight = _entity_weight(entity_freq[entity], weight_scheme)

        # Compute bigram counts for the entity's role sequence across sentences
        for k in range(len(roles_seq) - 1):
            bigram_counts[(roles_seq[k], roles_seq[k + 1])] += weight

        # Compute trigram counts for the entity's role sequence across sentences if trigrams are being used and there are enough sentences
        if use_trigrams and len(roles_seq) >= 3:
            for k in range(len(roles_seq) - 2):
                trigram_counts[(roles_seq[k], roles_seq[k + 1], roles_seq[k + 2])] += weight

    # No entity survived the frequency filter -> no usable profile
    bigram_total = sum(bigram_counts.values())
    if bigram_total == 0:
        return None, None, grid, sents, entity_freq

    bigram_profile = np.array([bigram_counts[t] / bigram_total for t in BIGRAM_TRANSITIONS])

    # Compute the trigram profile if trigrams are being used and there are enough sentences
    trigram_profile = None
    if use_trigrams:
        trigram_total = sum(trigram_counts.values())

        # Only compute the trigram profile if there are surviving trigram counts
        if trigram_total > 0:
            trigram_profile = np.array([trigram_counts[t] / trigram_total for t in TRIGRAM_TRANSITIONS])

    return bigram_profile, trigram_profile, grid, sents, entity_freq


def build_binary_grid(text, nlp=None):
    """Binary presence/absence entity grid for one text.

    Returns (grid, sents):
      grid: {entity_name: [0/1, 0/1, ...]}, presence per sentence
      sents: list of sentence spans from the single parse (alpha-only, no
        minimum-length threshold)
    """

    # Parse the text once; entity extraction runs directly on the resulting
    # sentence spans (no second parse over re-joined sentence strings)
    nlp = nlp if nlp is not None else get_spacy_nlp()
    doc = nlp(text)
    sents = [s for s in doc.sents if any(ch.isalpha() for ch in s.text)]

    # If there are fewer than 2 sentences, return empty structures since we can't compute transitions
    if len(sents) < 2:
        return {}, sents

    sent_entities = [extract_entities_simple(s) for s in sents]

    # Collect all unique entities across sentences to build the grid
    all_entities = set()
    for ents in sent_entities:
        all_entities.update(ents)
    if not all_entities:
        return {}, sents

    # For each entity, create a binary presence/absence sequence across sentences
    grid = {
        entity: [1 if entity in ents else 0 for ents in sent_entities]
        for entity in all_entities
    }

    return grid, sents


def compute_binary_transition_profile(text, nlp=None):
    """Presence/absence transition profile (4 bins: 00, 01, 10, 11) for one text.
    Entities are weighted equally, with no salience weighting or frequency filtering.

    Returns None when the text has no usable entity profile (< 2 sentences or no entities).
    """
    # Use spaCy to build the binary entity grid and extract sentences
    grid, sents = build_binary_grid(text, nlp=nlp)

    # No entities / too few sentences -> no usable profile (NA, not a uniform fallback)
    if not grid or len(sents) < 2:
        return None

    # Compute presence/absence transition counts across all entities
    counts = Counter()
    for presence in grid.values():
        for k in range(len(presence) - 1):
            counts[(presence[k], presence[k + 1])] += 1

    # No entity survived the frequency filter -> no usable profile
    total = sum(counts.values())
    if total == 0:
        return None

    # Normalize the counts to create a probability profile over the 4 transition types
    return np.array([counts[t] / total for t in BINARY_TRANSITIONS])


def jsd(p, q):
    """Jensen-Shannon divergence (symmetric, in [0, ln2]).

    Both inputs must be non-negative and carry positive mass
    """
    # Convert inputs to numpy arrays and normalize them to form valid probability distributions
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)

    p = p / p.sum()
    q = q / q.sum()

    # Compute the Jensen-Shannon divergence using the average distribution m
    m = 0.5 * (p + q)
    return 0.5 * (entropy(p, m) + entropy(q, m))


def discourse_diversity(texts, use_trigrams=True, alpha_trigram=0.3, nlp=None, weight_scheme=None):
    """D_disc for one group of texts: mean pairwise JSD of their transition profiles.

    D_disc = (1 - alpha_trigram) * JSD_bigram + alpha_trigram * JSD_trigram

    Texts with no usable entity profile are dropped. The bigram term averages
    over all remaining texts. The trigram term averages over the subset with a
    trigram profile (>= 3 sentences) and is only added when at least two such
    texts remain. Returns np.nan if fewer than two texts have a usable profile.
    """
    # At least 2 texts are required to compute pairwise diversity
    if len(texts) < 2:
        raise ValueError("discourse_diversity requires at least 2 texts")

    # Compute the transition profiles (bigram and trigram) for each text in the group, using spaCy for NLP processing
    profiles_bi, profiles_tri = [], []
    for t in texts:
        bi, tri, _, _, _ = compute_transition_profile(
            t, use_trigrams=use_trigrams, nlp=nlp, weight_scheme=weight_scheme
        )

        # Only keep texts with a usable bigram profile (>= 2 sentences, >= 1 entity passing the frequency filter)
        if bi is None:
            continue
        profiles_bi.append(bi)

        # Only keep texts with a usable trigram profile (>= 3 sentences, >= 1 entity passing the frequency filter)
        if tri is not None:
            profiles_tri.append(tri)

    # Not enough usable profiles -> undefined
    if len(profiles_bi) < 2:
        return float("nan")

    # Compute the mean pairwise Jensen-Shannon divergence (JSD) for bigram profiles across all pairs of texts
    profiles_bi = np.array(profiles_bi)
    bi_pairs = list(combinations(range(len(profiles_bi)), 2))
    jsd_bi = np.mean([jsd(profiles_bi[i], profiles_bi[j]) for i, j in bi_pairs])

    # If trigrams are being used and at least two texts have a trigram profile, compute the mean
    # pairwise trigram JSD over that subset and combine it with the bigram JSD via alpha_trigram
    if use_trigrams and len(profiles_tri) >= 2:
        profiles_tri = np.array(profiles_tri)
        tri_pairs = list(combinations(range(len(profiles_tri)), 2))
        jsd_tri = np.mean([jsd(profiles_tri[i], profiles_tri[j]) for i, j in tri_pairs])
        return float((1 - alpha_trigram) * jsd_bi + alpha_trigram * jsd_tri)

    return float(jsd_bi)


def discourse_diversity_simple(texts, nlp=None):
    """D_disc (simplified) for one group of texts: mean pairwise JSD of binary
    presence/absence transition profiles.

    Entities are weighted equally, without frequency filtering or grammatical
    roles. Texts with no usable entity profile are dropped. Returns np.nan if
    fewer than two texts have a usable profile.
    """
    # At least 2 texts are required to compute pairwise diversity
    if len(texts) < 2:
        raise ValueError("discourse_diversity_simple requires at least 2 texts")

    # Compute the binary transition profile for each text in the group, using spaCy for NLP processing
    nlp = nlp if nlp is not None else get_spacy_nlp()
    profiles = [
        p for p in (compute_binary_transition_profile(t, nlp=nlp) for t in texts) if p is not None
    ]

    # Not enough usable profiles -> undefined
    if len(profiles) < 2:
        return float("nan")

    # Compute the mean pairwise Jensen-Shannon divergence (JSD) across all pairs of texts
    profiles = np.array(profiles)
    pairs = list(combinations(range(len(profiles)), 2))
    
    return float(np.mean([jsd(profiles[i], profiles[j]) for i, j in pairs]))
