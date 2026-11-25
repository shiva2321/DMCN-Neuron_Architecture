"""Resonant Predictive Neuron (RPN) Word Formation & Morphological Generalization System.

Implements dual-track character orthotactics and morphological role-filler binding
using Vector Symbolic Architectures (VSA / phasor substrate) and fast-weight DeltaMemory.

Core Properties:
1. Multi-scale phasor n-gram context: Superimposes permutation-bound bigram
   and trigram conjunctions into a single normalized complex phasor.
2. Fast-weight delta updates: Memory writes residual error (LMS / delta rule) avoiding
   catastrophic saturation over hundreds of words.
3. Rectified Born gate & Refusal: Refuses to guess on phonotactically illegal prefixes
   (anti-hallucination property).
4. Morphological Generalization: Binds inflectional features (FEAT_ING, FEAT_ED, FEAT_S, FEAT_ER)
   at morpheme boundaries, enabling zero-shot inflection on completely novel pseudo-words (Wug test).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np

from ..substrate.phasor import (
    random_hv,
    permute,
    normalise,
    sim,
    sim_batch,
    bind,
    bundle,
)
from .delta_memory import DeltaMemory
from .born_gate import rectified_born


# 500 Curated common English words of lengths 4-8
VOCAB_500: List[str] = [
    "able", "acid", "aged", "also", "area", "army", "away", "baby", "back", "ball",
    "band", "bank", "base", "bath", "bear", "beat", "been", "beer", "bell", "belt",
    "best", "bill", "bird", "blow", "blue", "boat", "body", "bomb", "bond", "bone",
    "book", "boom", "born", "boss", "both", "bowl", "bulk", "burn", "bush", "busy",
    "call", "calm", "came", "camp", "card", "care", "case", "cash", "cast", "cell",
    "chat", "chip", "city", "club", "coal", "coat", "code", "cold", "come", "cook",
    "cool", "cope", "copy", "core", "cost", "crew", "crop", "dark", "data", "date",
    "dawn", "days", "dead", "deal", "dean", "dear", "debt", "deep", "deny", "desk",
    "dial", "diet", "disc", "disk", "does", "done", "door", "dose", "down", "draw",
    "drop", "drug", "dual", "duke", "dust", "duty", "each", "earn", "ease", "east",
    "easy", "edge", "else", "even", "ever", "face", "fact", "fail", "fair", "fall",
    "farm", "fast", "fate", "fear", "feed", "feel", "feet", "fell", "file", "fill",
    "film", "find", "fine", "fire", "firm", "fish", "five", "flat", "flow", "food",
    "foot", "ford", "form", "fort", "four", "free", "from", "fuel", "full", "fund",
    "gain", "game", "gate", "gave", "gear", "gift", "girl", "give", "glad", "goal",
    "goes", "gold", "golf", "gone", "good", "gray", "grew", "grey", "grow", "gulf",
    "hair", "half", "hall", "hand", "hang", "hard", "harm", "hate", "have", "head",
    "hear", "heat", "held", "hell", "help", "here", "hero", "high", "hill", "hire",
    "hold", "hole", "holy", "home", "hope", "host", "hour", "huge", "hung", "hunt",
    "hurt", "idea", "inch", "into", "iron", "item", "jack", "jane", "jean", "john",
    "join", "jump", "jury", "just", "keen", "keep", "kept", "kick", "kill", "kind",
    "king", "knee", "knew", "know", "lack", "lady", "laid", "lake", "land", "lane",
    "last", "late", "lead", "left", "less", "life", "lift", "like", "line", "link",
    "list", "live", "load", "loan", "lock", "logo", "long", "look", "lord", "lose",
    "loss", "lost", "love", "luck", "made", "mail", "main", "make", "male", "many",
    "mark", "mass", "matt", "meal", "mean", "meat", "meet", "menu", "mere", "mike",
    "mile", "milk", "mill", "mind", "mine", "miss", "mode", "mood", "moon", "more",
    "most", "move", "much", "must", "name", "navy", "near", "neck", "need", "news",
    "next", "nice", "nick", "nine", "none", "nose", "note", "okay", "once", "only",
    "onto", "open", "oral", "over", "pace", "pack", "page", "paid", "pain", "pair",
    "palm", "park", "part", "pass", "past", "path", "peak", "peer", "pick", "pile",
    "pink", "pipe", "plan", "play", "plot", "plug", "plus", "poll", "pool", "poor",
    "port", "post", "pull", "pure", "push", "race", "rail", "rain", "rank", "rare",
    "rate", "read", "real", "rear", "rely", "rent", "rest", "rice", "rich", "ride",
    "ring", "rise", "risk", "road", "rock", "role", "roll", "roof", "room", "root",
    "rose", "rule", "rush", "safe", "said", "sake", "sale", "salt", "same", "sand",
    "save", "seat", "seed", "seek", "seem", "seen", "self", "sell", "send", "sent",
    "sept", "ship", "shop", "shot", "show", "shut", "sick", "side", "sign", "site",
    "size", "skin", "slip", "slow", "snow", "soft", "soil", "sold", "sole", "some",
    "song", "soon", "sort", "soul", "spot", "star", "stay", "step", "stop", "such",
    "suit", "sure", "take", "tale", "talk", "tall", "tank", "tape", "task", "team",
    "tech", "tell", "tend", "term", "test", "text", "than", "that", "them", "then",
    "they", "thin", "this", "thus", "tide", "tied", "tier", "till", "time", "tiny",
    "told", "toll", "tone", "tony", "took", "tool", "tour", "town", "tree", "trip",
    "true", "tune", "turn", "twin", "type", "unit", "upon", "used", "user", "vary",
    "vast", "very", "vice", "view", "vote", "wage", "wait", "wake", "walk", "wall",
    "want", "ward", "warm", "wash", "wave", "ways", "weak", "wear", "week", "well",
    "went", "were", "west", "what", "when", "whom", "wide", "wife", "wild", "will",
    "wind", "wine", "wing", "wire", "wise", "wish", "with", "wood", "word", "wore",
    "work", "yard", "yeah", "year", "your", "zero", "zinc", "zone"
]

# Morphological Regular Verbs with Systematic Inflections
INFLECTED_BASE_VERBS: List[str] = [
    "walk", "play", "hunt", "jump", "farm", "cook", "clean", "paint",
    "call", "pack", "track", "print", "test", "plant", "help", "start",
    "count", "watch", "reach", "open", "cross", "dress", "pass", "fish",
    "wash", "wish", "mark", "point", "sound", "light", "turn", "wait"
]

# 20 Novel pseudo-words for the Wug Test (NEVER seen in English or training)
WUG_STEMS: List[str] = [
    "wug", "plap", "blik", "snool", "crad", "flep", "glop", "stran",
    "twik", "spod", "vrek", "snarp", "plim", "brast", "crong", "florp",
    "skell", "grisp", "dret", "klom"
]

# Phonotactically Illegal Prefixes (Consonant clusters impossible in English onsets)
ILLEGAL_PREFIXES: List[str] = [
    "^qzk", "^xjv", "^bzp", "^ptk", "^fgj", "^vxq", "^zkm", "^jpq", "^rkp", "^wql",
    "^cbk", "^dxz", "^fkq", "^gxk", "^jxp", "^kqx", "^lxz", "^mxk", "^nxp", "^pxk"
]

# Canonical Legal English Onset Prefixes
LEGAL_PREFIXES: List[str] = [
    "^st", "^tr", "^pl", "^ch", "^sh", "^br", "^fl", "^cl", "^pr", "^gr",
    "^bl", "^cr", "^dr", "^fr", "^gl", "^sc", "^sk", "^sl", "^sm", "^sn"
]

# Morphological Features & Suffix Mappings
MORPH_FEATURES = {
    "FEAT_BASE": "",
    "FEAT_ING": "ing",
    "FEAT_ED": "ed",
    "FEAT_S": "s",
    "FEAT_ER": "er",
}


class PhasorLexicon:
    """Stable unit phasor mapping for characters, boundary tokens, and morphological features."""

    def __init__(self, D: int = 512, seed: int = 42):
        self.D = D
        self.seed = seed
        self.chars = [chr(i) for i in range(ord('a'), ord('z') + 1)] + ['^', '$', '+']
        self.features = list(MORPH_FEATURES.keys())
        
        # Build deterministic pseudo-random unit phasors
        self.char_hvs: Dict[str, np.ndarray] = {
            c: random_hv(D, seed=seed * 1000 + i) for i, c in enumerate(self.chars)
        }
        self.feat_hvs: Dict[str, np.ndarray] = {
            f: random_hv(D, seed=seed * 2000 + i) for i, f in enumerate(self.features)
        }
        
        self.char_matrix = np.stack([self.char_hvs[c] for c in self.chars]).astype(np.complex64)


class RPNWordFormationSystem:
    """Resonant Predictive Neuron Word Formation & Generalization Architecture."""

    def __init__(
        self,
        D: int = 512,
        seed: int = 42,
        lr: float = 0.5,
        confidence_threshold: float = 0.35,
        min_resonance: float = 0.30,
    ):
        self.D = D
        self.seed = seed
        self.lr = float(lr)
        self.confidence_threshold = float(confidence_threshold)
        self.min_resonance = float(min_resonance)
        
        self.lexicon = PhasorLexicon(D=D, seed=seed)
        self.mem = DeltaMemory(D, mode="delta", lr=lr)
        self.affix_mem = DeltaMemory(D, mode="delta", lr=lr)

    def encode_orthotactic_context(
        self,
        prefix: str,
        feature: Optional[str] = None,
    ) -> np.ndarray:
        """Encode multi-scale n-gram context: permutation-bound bigram and trigram conjunctions."""
        if not prefix:
            return np.zeros(self.D, dtype=np.complex64)
            
        components: List[np.ndarray] = []
        if feature is not None and feature in self.lexicon.feat_hvs:
            components.append(self.lexicon.feat_hvs[feature])
            
        if len(prefix) == 1:
            components.append(self.lexicon.char_hvs[prefix[0]])
            return bundle(*components, normalise_output=True)
            
        c1 = self.lexicon.char_hvs[prefix[-1]]
        c2 = self.lexicon.char_hvs[prefix[-2]]
        
        # Bigram conjunction: rho(c2) * c1
        bigram = bind(permute(c2, 1), c1)
        components.append(bigram)
        
        # Trigram conjunction: rho^2(c3) * rho(c2) * c1
        if len(prefix) >= 3:
            c3 = self.lexicon.char_hvs[prefix[-3]]
            trigram = bind(bind(permute(c3, 2), permute(c2, 1)), c1)
            components.append(trigram)
            
        return bundle(*components, normalise_output=True)

    def learn_word(
        self,
        word: str,
        feature: str = "FEAT_BASE",
    ) -> None:
        """Store character transitions of a word in fast-weight DeltaMemory."""
        formatted = f"^{word}$" if not word.startswith("^") else word
        for t in range(1, len(formatted)):
            ctx = self.encode_orthotactic_context(formatted[:t], feature=feature)
            target = self.lexicon.char_hvs[formatted[t]]
            self.mem.write(ctx, target)

    def learn_inflection(
        self,
        stem: str,
        feature: str,
    ) -> None:
        """Learn morphological inflection at the morpheme boundary (+)."""
        suffix = MORPH_FEATURES[feature]
        full_word = f"^{stem}+{suffix}$" if suffix else f"^{stem}+$"
        
        # 1. Stem orthotactics in base memory
        for t in range(1, len(stem) + 1):
            ctx = self.encode_orthotactic_context(full_word[:t])
            target = self.lexicon.char_hvs[full_word[t]]
            self.mem.write(ctx, target)
            
        # 2. Affix sequence in morphological memory
        boundary_idx = len(stem) + 1
        for t in range(boundary_idx, len(full_word)):
            ctx = self.encode_orthotactic_context(full_word[:t], feature=feature)
            target = self.lexicon.char_hvs[full_word[t]]
            self.affix_mem.write(ctx, target)

    def train_corpus(
        self,
        corpus_words: Sequence[str],
        inflected_verbs: Sequence[str] = INFLECTED_BASE_VERBS,
    ) -> None:
        """Train the system on hundreds of vocabulary words and inflectional paradigms."""
        # 1. General vocabulary
        for w in corpus_words:
            self.learn_word(w, feature="FEAT_BASE")
            
        # 2. Morphological inflection paradigms
        for v in inflected_verbs:
            for feat in MORPH_FEATURES.keys():
                self.learn_inflection(v, feat)

    def predict_next(
        self,
        prefix: str,
        feature: Optional[str] = None,
        use_affix_mem: bool = False,
    ) -> Tuple[Optional[str], float, np.ndarray]:
        """Predict the next character with Born gate confidence/resonance refusal."""
        active_mem = self.affix_mem if use_affix_mem else self.mem
        ctx = self.encode_orthotactic_context(prefix, feature=feature)
        pred = active_mem.predict(ctx)
        
        confidence = float(np.linalg.norm(pred)) / np.sqrt(self.D)
        if confidence < self.confidence_threshold:
            return None, 0.0, np.zeros(len(self.lexicon.chars), dtype=np.float32)
            
        echo = normalise(pred)
        sims = sim_batch(self.lexicon.char_matrix, echo)
        
        gate_out = rectified_born(
            sims,
            gate_threshold=0.05,
            min_resonance=self.min_resonance,
        )
        if not gate_out.gate_passed:
            return None, float(gate_out.best_prob), sims
            
        return self.lexicon.chars[gate_out.best_idx], float(gate_out.best_prob), sims

    def complete_word(
        self,
        prompt: str,
        max_steps: int = 8,
        feature: Optional[str] = None,
    ) -> Tuple[str, bool]:
        """Autoregressively complete a word from a prefix until '$' or refusal."""
        curr = f"^{prompt}" if not prompt.startswith("^") else prompt
        finished = False
        for _ in range(max_steps):
            next_c, prob, _ = self.predict_next(curr, feature=feature)
            if next_c is None:
                break
            if next_c == '$':
                finished = True
                break
            curr += next_c
            
        completed = curr[1:]  # strip leading '^'
        return completed, finished

    def inflect_wug(
        self,
        stem: str,
        feature: str,
        max_steps: int = 6,
    ) -> str:
        """Autonomously generate the inflected form for a novel unseen Wug stem."""
        curr = f"^{stem}+"
        for _ in range(max_steps):
            next_c, prob, _ = self.predict_next(curr, feature=feature, use_affix_mem=True)
            if next_c is None or next_c == '$':
                break
            curr += next_c
        return curr[1:].replace("+", "")

    def evaluate_held_out(
        self,
        test_words: Sequence[str],
    ) -> Dict[str, float]:
        """Evaluate next-character Top-1 and Top-3 accuracy on held-out unseen words."""
        top1_correct = 0
        top3_correct = 0
        total_transitions = 0
        
        for w in test_words:
            formatted = f"^{w}$" if not w.startswith("^") else w
            for t in range(1, len(formatted)):
                prefix = formatted[:t]
                expected = formatted[t]
                ctx = self.encode_orthotactic_context(prefix)
                pred = self.mem.predict(ctx)
                echo = normalise(pred)
                sims = sim_batch(self.lexicon.char_matrix, echo)
                
                top3_indices = np.argsort(sims)[-3:]
                top1_char = self.lexicon.chars[np.argmax(sims)]
                top3_chars = [self.lexicon.chars[idx] for idx in top3_indices]
                
                if top1_char == expected:
                    top1_correct += 1
                if expected in top3_chars:
                    top3_correct += 1
                total_transitions += 1
                
        return {
            "top1_acc": float(100.0 * top1_correct / max(1, total_transitions)),
            "top3_acc": float(100.0 * top3_correct / max(1, total_transitions)),
            "total_transitions": total_transitions,
        }

    def evaluate_completion(
        self,
        test_words: Sequence[str],
        vocabulary_set: set[str],
        prompt_len: int = 2,
    ) -> Dict[str, float]:
        """Evaluate word completion from 2-letter prompts on held-out words."""
        valid_words = 0
        total = 0
        for w in test_words:
            if len(w) <= prompt_len:
                continue
            prompt = w[:prompt_len]
            completed, finished = self.complete_word(prompt)
            if completed in vocabulary_set:
                valid_words += 1
            total += 1
            
        return {
            "completion_validity_rate": float(100.0 * valid_words / max(1, total)),
            "evaluated_words": total,
        }

    def evaluate_refusal(
        self,
        illegal_prefixes: Sequence[str] = ILLEGAL_PREFIXES,
        legal_prefixes: Sequence[str] = LEGAL_PREFIXES,
    ) -> Dict[str, float]:
        """Evaluate Born gate anti-hallucination refusal on illegal vs legal prefixes."""
        refused_illegal = 0
        for p in illegal_prefixes:
            next_c, prob, _ = self.predict_next(p)
            if next_c is None:
                refused_illegal += 1
                
        refused_legal = 0
        for p in legal_prefixes:
            next_c, prob, _ = self.predict_next(p)
            if next_c is None:
                refused_legal += 1
                
        return {
            "illegal_refusal_rate": float(100.0 * refused_illegal / max(1, len(illegal_prefixes))),
            "legal_refusal_rate": float(100.0 * refused_legal / max(1, len(legal_prefixes))),
        }

    def evaluate_wug_generalization(
        self,
        stems: Sequence[str] = WUG_STEMS,
    ) -> Dict[str, Any]:
        """Evaluate zero-shot morphological inflection on novel unseen Wug stems."""
        correct = 0
        total = 0
        details: List[Dict[str, Any]] = []
        
        test_features = ["FEAT_ING", "FEAT_ED", "FEAT_S", "FEAT_ER"]
        for stem in stems:
            for feat in test_features:
                expected = stem + MORPH_FEATURES[feat]
                generated = self.inflect_wug(stem, feat)
                is_correct = (generated == expected)
                if is_correct:
                    correct += 1
                total += 1
                details.append({
                    "stem": stem,
                    "feat": feat,
                    "expected": expected,
                    "generated": generated,
                    "correct": is_correct,
                })
                
        return {
            "wug_accuracy": float(100.0 * correct / max(1, total)),
            "correct_count": correct,
            "total_queries": total,
            "details": details,
        }
