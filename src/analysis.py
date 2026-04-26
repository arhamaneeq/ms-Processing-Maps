from src.parser import parse
import matplotlib.pyplot as plt

ALLOYS = parse()

# print(ALLOYS['crno'])


for ALLOY in ALLOYS:
    print(ALLOY)