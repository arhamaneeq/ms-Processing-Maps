from src.parser import parse
import matplotlib.pyplot as plt

ALLOYS = parse()

for ALLOY, DFS in ALLOYS.items():
    for STRATEGY, DF in DFS.items():
        print(f"{ALLOY} {STRATEGY}")
        print(DF.head())