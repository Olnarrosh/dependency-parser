class Token:
    """Represents a token in a sentence. Contains the word form, lemma, part of
    speech, the position of its head, and the type of dependency relation.
    """
    def __init__(self, form: str, head: int, relation: str, lemma: str = None, pos: str = None):
        self.form = form
        self.head = head
        self.relation = relation
        self.lemma = lemma or "(none)"
        self.pos = pos or "(NONE)"

class Sentence:
    """Represents a sentence as a sequence of tokens, and a mapping of potential
    heads and dependents (as integer tuples) to features.
    """
    def __init__(self, source = None):
        if source:
            self.tokens = [Token(t.form, None, None, t.lemma, t.pos) for t in source.tokens]
        else:
            self.tokens = [Token("ROOT", None, None, pos="(ROOT)")]
        self.features = {}

    def __len__(self) -> int:
        return len(self.tokens)

    def __getitem__(self, n: int) -> Token:
        return self.tokens[n]
    
    def append(self, token: Token) -> None:
        self.tokens.append(token)
