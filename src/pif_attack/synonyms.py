"""Synonym generation using WordNet."""
from nltk.corpus import wordnet as wn
import random

class SynonymGenerator:
    def __init__(self, max_candidates=5):
        self.max_candidates = max_candidates
    
    def get_synonyms(self, word):
        """Get synonyms for a word."""
        synonyms = []
        
        try:
            for syn in wn.synsets(word):
                for lemma in syn.lemmas():
                    synonym = lemma.name().replace("_", " ")
                    if synonym.lower() != word.lower() and synonym not in synonyms:
                        synonyms.append(synonym)
        except:
            pass
        
        # If no synonyms found, try simple variations
        if not synonyms:
            synonyms = [word.capitalize(), word.upper(), word + "s"]
        
        return synonyms[:self.max_candidates]
