"""
Discourse diversity (D_disc): entity-grid-based diversity metric

Full entity-grid model from Barzilay & Lapata (2008): 
sentences -> per-entity grammatical roles (Subject / Object / Other / Absent) 
-> weighted bigram + trigram role-transition profiles 
-> mean pairwise Jensen-Shannon divergence across a group of texts (e.g. multiple generations for the same prompt).

    from metrics_lib.discourse import discourse_diversity
    d_disc = discourse_diversity(list_of_texts_for_one_prompt)
"""
import re
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

MIN_ENTITY_FREQ = 2  # entity must appear in >= this many sentences to count

# Grammatical dependency labels for subjects and objects (spaCy)
_SUBJ_DEPS = {"nsubj", "nsubjpass", "csubj", "csubjpass"}
_OBJ_DEPS = {"dobj", "iobj", "pobj", "attr", "oprd", "dative"}
_VALID_NER = {"PERSON", "ORG", "GPE", "LOC", "PRODUCT", "EVENT", "WORK_OF_ART", "FAC", "NORP"}



def sent_tokenize(text, nlp=None):
    """Sentence-tokenize using spaCy's sentencizer"""
    nlp = nlp if nlp is not None else get_spacy_nlp()
    sents = [sent.text.strip() for sent in nlp(text).sents]

    return [s for s in sents if len(s.strip()) > 5]


def _get_role_for_token(token):
    """Grammatical role of a token via its own or its head's dependency label."""
    if token.dep_ in _SUBJ_DEPS:
        return ROLE_S
    if token.dep_ in _OBJ_DEPS:
        return ROLE_O
    if token.head != token:
        if token.head.dep_ in _SUBJ_DEPS:
            return ROLE_S
        if token.head.dep_ in _OBJ_DEPS:
            return ROLE_O
    return ROLE_X


def extract_entity_roles_spacy(sent_doc):
    """{entity_name: role} for one spaCy-parsed sentence. Ties broken S > O > X."""
    entity_roles = {}
    role_priority = {ROLE_S: 3, ROLE_O: 2, ROLE_X: 1}

    # Use spaCy's named entity recognition to find entities and assign roles based on grammatical dependencies
    for ent in sent_doc.ents:
        # Ignore entities that are not of the valid types (PERSON, ORG, GPE, etc.)
        if ent.label_ not in _VALID_NER:
            continue

        # Normalize entity name to lowercase and strip whitespace; skip if too short
        name = ent.text.lower().strip()
        if len(name) < 2:
            continue

        # Determine the role of the entity based on its tokens and their dependencies, prioritizing Subject > Object > Other
        role = ROLE_X
        for token in ent:
            token_role = _get_role_for_token(token)
            if role_priority.get(token_role, 0) > role_priority.get(role, 0):
                role = token_role

        # If the entity is already in the dictionary, only update its role if the new role has higher priority (S > O > X)
        if name not in entity_roles or role_priority[role] > role_priority[entity_roles[name]]:
            entity_roles[name] = role

    return entity_roles




def build_entity_grid(text, nlp=None):
    """Entity grid for one text.

    Returns (grid, sents, entity_freq):
      grid: {entity_name: [role_sent1, role_sent2, ...]}, role in {S, O, X, -}
      sents: list of sentences
      entity_freq: {entity_name: number of sentences it appears in}
    """

    # Use spaCy to sentence-tokenize and extract entity roles for each sentence
    nlp = nlp if nlp is not None else get_spacy_nlp()
    sents = sent_tokenize(text, nlp=nlp)

    # If there are fewer than 2 sentences, return empty structures since we can't compute transitions
    if len(sents) < 2:
        return {}, sents, {}

    # Use spaCy's nlp.pipe to efficiently process all sentences in batches and extract entity roles for each sentence
    docs = list(nlp.pipe(sents, batch_size=50))
    sent_entity_roles = [extract_entity_roles_spacy(doc) for doc in docs]

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


def compute_transition_profile(text, use_trigrams=True, nlp=None):
    """Weighted bigram/trigram role-transition profile for one text.

    - Salience weighting w_e = log(1 + freq(e))
    - Entities below MIN_ENTITY_FREQ are dropped
    
    Returns (bigram_profile, trigram_profile_or_None, grid, sents, entity_freq)
    """
    # Use spaCy to build the entity grid and extract sentences and entity frequencies
    grid, sents, entity_freq = build_entity_grid(text, nlp=nlp)

    # If there are no entities or fewer than 2 sentences, return uniform distributions for bigram and trigram profiles
    if not grid or len(sents) < 2:
        return (
            np.ones(N_BIGRAM) / N_BIGRAM,
            np.ones(N_TRIGRAM) / N_TRIGRAM if use_trigrams else None,
            grid, sents, entity_freq,
        )

    # Compute weighted bigram and trigram counts for the entity roles across sentences, applying salience weighting based on entity frequency
    bigram_counts, trigram_counts = Counter(), Counter()
    for entity, roles_seq in grid.items():
        if entity_freq[entity] < MIN_ENTITY_FREQ:
            continue
        # Compute the salience weight for the entity based on its frequency across sentences
        weight = np.log(1 + entity_freq[entity])

        # Compute bigram counts for the entity's role sequence across sentences
        for k in range(len(roles_seq) - 1):
            bigram_counts[(roles_seq[k], roles_seq[k + 1])] += weight

        # Compute trigram counts for the entity's role sequence across sentences if trigrams are being used and there are enough sentences
        if use_trigrams and len(roles_seq) >= 3:
            for k in range(len(roles_seq) - 2):
                trigram_counts[(roles_seq[k], roles_seq[k + 1], roles_seq[k + 2])] += weight

    # Normalize the bigram and trigram counts to create probability profiles for the entity role transitions, ensuring that the total counts are not zero to avoid division by zero errors
    bigram_total = sum(bigram_counts.values()) or 1
    bigram_profile = np.array([bigram_counts[t] / bigram_total for t in BIGRAM_TRANSITIONS])

    trigram_profile = None
    if use_trigrams:
        trigram_total = sum(trigram_counts.values()) or 1
        trigram_profile = np.array([trigram_counts[t] / trigram_total for t in TRIGRAM_TRANSITIONS])

    return bigram_profile, trigram_profile, grid, sents, entity_freq


def jsd(p, q):
    """Jensen-Shannon divergence (symmetric, in [0, ln2])."""
    # Add a small constant to avoid division by zero
    p = np.asarray(p, dtype=float) + 1e-10
    q = np.asarray(q, dtype=float) + 1e-10

    # Normalize the distributions to ensure they sum to 1
    p /= p.sum()
    q /= q.sum()

    # Compute the average distribution and calculate the Jensen-Shannon divergence using the entropy function
    m = 0.5 * (p + q)
    return 0.5 * (entropy(p, m) + entropy(q, m))


def discourse_diversity(texts, use_trigrams=True, alpha_trigram=0.3, nlp=None):
    """D_disc for one group of texts: mean pairwise JSD of their transition profiles.

    D_disc = (1 - alpha_trigram) * JSD_bigram + alpha_trigram * JSD_trigram
    """
    # At least 2 texts are required to compute pairwise diversity
    if len(texts) < 2:
        raise ValueError("discourse_diversity requires at least 2 texts")

    # Compute the transition profiles (bigram and trigram) for each text in the group, using spaCy for NLP processing
    profiles_bi, profiles_tri = [], []
    for t in texts:
        bi, tri, _, _, _ = compute_transition_profile(t, use_trigrams=use_trigrams, nlp=nlp)
        profiles_bi.append(bi)
        if tri is not None:
            profiles_tri.append(tri)

    # Compute the mean pairwise Jensen-Shannon divergence (JSD) for bigram profiles across all pairs of texts
    profiles_bi = np.array(profiles_bi)
    pairs = list(combinations(range(len(texts)), 2))
    jsd_bi = np.mean([jsd(profiles_bi[i], profiles_bi[j]) for i, j in pairs])

    # If trigrams are being used and trigram profiles were successfully computed, compute the mean pairwise JSD 
    # for trigram profiles and combine it with the bigram JSD using the specified alpha_trigram weight.
    if use_trigrams and profiles_tri:
        profiles_tri = np.array(profiles_tri)
        jsd_tri = np.mean([jsd(profiles_tri[i], profiles_tri[j]) for i, j in pairs])
        return float((1 - alpha_trigram) * jsd_bi + alpha_trigram * jsd_tri)

    return float(jsd_bi)
