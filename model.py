from graph import Graph
from corpus import Sentence, Token
from math import inf, log

class Model:
    def __init__(self):
        self.weights_pos = {}
        self.weights_neg = {}
        self.ratio = 1

    def train(self, corpus: list[Sentence]) -> None:
        for s in corpus:
            for h in range(len(s)):
                for d in range(1, len(s)):
                    if h != d:
                        self._train_(s, h, d)
        self.ratio = sum(sum(w.values()) for w in self.weights_neg.values()) / sum(sum(w.values()) for w in self.weights_pos.values())
        
    def _train_(self, s: Sentence, h: int, d: int) -> None:
        for f in s.features[(h, d)]:
            weights = self.weights_pos if s[d].head == h else self.weights_neg
            if not f in weights:
                weights[f] = {}
            weights[f][s[d].relation] = weights[f].get(s[d].relation, 0) + 1

    def predict(self, s: Sentence) -> Sentence:
        prediction = Sentence(s)
        edges = []
        for h in range(len(s)):
            for d in range(1, len(s)):
                if h == d:
                    continue
                weights = {}
                for f in s.features[(h, d)]:
                    for label in self.weights_neg.get(f, []):
                        weights[label] = weights.get(label, 0) + log(self.weights_neg[f][label])
                    for label in self.weights_pos.get(f, []):
                        weights[label] = weights.get(label, 0) - log(self.weights_pos[f][label] * self.ratio)
                label, value = min(weights.items(), key=lambda p: p[1]) if weights else ("", inf)
                edges.append((h, d, value, label))
        tree = Graph(len(s), edges).cle()
        for edge in tree:
            prediction[edge.target].head = edge.origin
            prediction[edge.target].relation = edge.label
        return prediction
    
    def test(self, corpus: list[Sentence]) -> dict:
        tok_unlabeled_correct = 0
        tok_labeled_correct = 0
        tok_total = 0
        sent_unlabeled_correct = 0
        sent_labeled_correct = 0
        sent_total = len(corpus)
        for s in corpus:
            incorrect_head = False
            incorrect_label = False
            prediction = self.predict(s)
            for i in range(1, len(s)):
                if prediction[i].head == s[i].head:
                    tok_unlabeled_correct += 1
                    if prediction[i].relation == s[i].relation:
                        tok_labeled_correct += 1
                    else:
                        incorrect_label = True
                else:
                    incorrect_head = True
                    incorrect_label = True
                tok_total += 1
            sent_unlabeled_correct += not incorrect_head
            sent_labeled_correct += not incorrect_label
        return {"UAS": tok_unlabeled_correct / tok_total,
                "LAS": tok_labeled_correct / tok_total,
                "UCM": sent_unlabeled_correct / sent_total,
                "LCM": sent_labeled_correct / sent_total}
