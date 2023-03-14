from corpus import Sentence

class FeatureExtractor:
    def __init__(self, templates = None):
        """Feature templates can be supplied as lists of lists of strings,
        which can be 'form', 'lem' (lemma), 'pos', 'pos-1', 'pos+1' (part
        of speech of the word in question or the previous or next word),
        each prefixed with 'd' (dependent) or 'h' (head), as well as 'bpos'
        (parts of speech of all words between head and dependent), 'dir'
        (direction) and 'dist' (distance).
        If the paramater is not supplied, a default combination is used.
        """
        self.string_map = {}
        self.templates = templates or [
            ("hlem", "hpos", "dlem", "dpos", "dir", "dist"),
            ("hlem", "hpos", "dlem", "dir", "dist"),
            ("hlem", "dlem", "dpos", "dir", "dist"),
            ("hlem", "hpos", "dpos", "dir", "dist"),
            ("hpos", "dlem", "dpos", "dir", "dist"),
            ("hpos", "dpos", "hpos+1", "dpos-1", "dir", "dist"),
            ("hpos", "dpos", "hpos-1", "dpos-1", "dir", "dist"),
            ("hpos", "dpos", "hpos+1", "dpos+1", "dir", "dist"),
            ("hpos", "dpos", "hpos-1", "dpos+1", "dir", "dist")
        ]

    def lookup(self, s: str, add_new: bool) -> int:
        """Look up a string in the string mapper. The string will only be
        added to the string map if the add_new parameter is true.
        If an unknown string is looked up, the next free non-negative
        integer is used.
        """
        return (self.string_map.setdefault if add_new else self.string_map.get)(s, len(self.string_map))

    def extract_features(self, corpus: list[Sentence], add_new: bool, templates: list[list[str]] = None) -> None:
        """Extract features from the corpus, either based on a given
        feature template, or the one used on construction of the object.
        The add_new parameter decides whether new entries are added to the
        string map in the process; it should be true for training data, and
        false for testing data.
        """
        for sentence in corpus:
            for h in range(len(sentence)):
                for d in range(len(sentence)):
                    if h != d and d != 0:
                        features = FeatureExtractor._extract_features_(sentence, h, d, templates or self.templates)
                        sentence.features[(h, d)] = sorted([self.lookup(f, add_new) for f in features])

    @staticmethod
    def _extract_features_(s: str, h: int, d: int, templates: list[list[str]]):
        features = {
            "hform": s[h].form.lower(),
            "hpos": s[h].pos,
            "hlem": s[h].lemma,
            "dform": s[d].form.lower(),
            "dpos": s[d].pos,
            "dlem": s[d].lemma,
            "bpos": "+".join(t.pos for t in s[min(h,d):max(h,d)-1]) or "(EMPTY)",
            "hpos+1": "(ROOT)" if h == 0 else (s[h+1].pos if h + 1 < len(s) else " "),
            "dpos+1": s[d+1].pos if d + 1 < len(s) else " ",
            "hpos-1": "(ROOT)" if h == 0 else (s[h-1].pos if h > 1 else " "),
            "dpos-1": s[d-1].pos if d > 1 else " ",
            "dir": "left" if h > d else "right",
            "dist": str(abs(h - d))
        }
        return map(lambda t: f"{','.join(t)}:{'+'.join(features[f] for f in t)}", templates)
