#!/usr/bin/python3
import clip
import torch
import sys
from pyparsing import *
import numpy as np

ppc = pyparsing_common

ParserElement.enablePackrat()
sys.setrecursionlimit(3000)

word = Word(alphas)
phrase = QuotedString("'", escChar='\\')
integer = ppc.integer

operand = word | phrase | integer
plusop = oneOf("+ -")
signop = oneOf("+ -")
multop = oneOf("* /")

expr = infixNotation(
    operand,
    [
        (multop, 2, opAssoc.LEFT),
        (plusop, 2, opAssoc.LEFT),
    ],
)


def text_concepts_to_vector(model, concepts):
    operator = ''
    features = []
    if len(concepts) != 3:
        raise 'unbalanced expression - expected <concept> <operator> <concept>'
    for concept in concepts:
        if type(concept) == str:
            if concept in ['+', '-', '/', '*']:
                operator = concept
            else:
                with torch.no_grad():
                    features.append(np.array(model.encode_text(clip.tokenize(concept))[0].tolist()))
        elif type(concept) == int:
            features.append(concept)
        else:
            features.append(np.array((text_concepts_to_vector(model, concept))))
    if operator == '+':
        return features[0] + features[1]
    elif operator == '-':
        return features[0] - features[1]
    elif operator == '/':
        return features[0] / features[1]
    elif operator == '*':
        return features[0] * features[1]
    else:
        raise f'unknown operator {operator}'


device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-L/14", device=device)

if __name__ == '__main__':
    for text in sys.stdin:
        concepts = expr.parseString(text.strip())
        print(text_concepts_to_vector(model, concepts[0]).tolist())
        sys.stdout.flush()
