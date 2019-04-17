import sys
sys.path.append("../")
from force_bdss_prototype.MCOwrapper import MCOwrapper

A = { "name": "eductA", "manufacturer": "", "pdi": 0 }
B = { "name": "eductB", "manufacturer": "", "pdi": 0 }
C = { "name": "contamination", "manufacturer": "", "pdi": 0 }
P = { "name": "product", "manufacturer": "", "pdi": 0 }
S = { "name": "sideproduct", "manufacturer": "", "pdi": 0 }
RP = { "reactants": [A, B], "products": [P] }
RS = { "reactants": [A, B], "products": [S] }

MCO = MCOwrapper(RP, C)
pp = MCO.solve()
