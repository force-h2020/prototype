from force_bdss_prototype.MCOwrapper import MCOwrapper

A = {"name": "eductA", "manufacturer": "", "pdi": 0}
B = {"name": "eductB", "manufacturer": "", "pdi": 0}
C = {"name": "contamination", "manufacturer": "", "pdi": 0}
P = {"name": "product", "manufacturer": "", "pdi": 0}
RP = {"reactants": [A, B], "products": [P]}

MCO = MCOwrapper(RP, C)
MCO.solve()
