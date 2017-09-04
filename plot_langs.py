#!/usr/bin/env pythonw

import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd

P_letter_lang = pd.read_csv('table_langs.dat', sep=' ', header=0, index_col = 0)

plt.plot(range(26), pd.np.array(P_letter_lang["eng"]), '-')
plt.plot(range(26), pd.np.array(P_letter_lang["fre"]), '-')
plt.plot(range(26), pd.np.array(P_letter_lang["ger"]), '-')
plt.plot(range(26), pd.np.array(P_letter_lang["ita"]), '-')
plt.plot(range(26), pd.np.array(P_letter_lang["spa"]), '-')
plt.xticks(list(range(26)), P_letter_lang.index)
plt.legend(["English", "French", "German", "Italian", "Spanish"])
plt.xlabel("letter")
plt.ylabel("P(letter)")
plt.savefig("letter_lang.png")