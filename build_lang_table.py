#!/usr/bin/env pythonw

import sys
import gzip
import string
import re
import numpy as np

letter_regex = re.compile(r'[a-z]', re.I)
characters = sorted(set(string.ascii_letters.lower()))

pos = dict(zip(characters, range(len(characters))))
counts = np.zeros(len(characters), dtype='uint64')

line_count = 0

for filename in sys.argv[1:]:
    for line in gzip.open(filename, "rt"):
        fields = line.lower().strip().split()

        line_count += 1

        if line_count % 100000 == 0:
            print(filename, line_count)
            break

        count = int(fields[2])
        word = fields[0]

        if "_" in word:
            continue

        letters = letter_regex.findall(word)

        if len(letters) != len(word):
            continue

        for letter in letters:
            if letter not in pos:
                continue

            counts[pos[letter]] += count

total = np.sum(counts)
pos = list(pos.items())
pos.sort(key=lambda x: x[1])

for key, value in enumerate(pos):
    print(value[0], counts[key]/total)