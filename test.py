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


print(expr.parseString("('germany' - berlin) + ('united kingdom' + bridge)"))
print(expr.parseString("('lion' + tiger) / 2"))

print(np.array([1,2,3]) + 2)