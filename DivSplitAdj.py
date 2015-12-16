import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import os

pd.set_option('notebook_repr_html', True)
pd.set_option('display.max_columns', 300)
pd.set_option('display.width', 3000)

folder = "data/DividendSplit/correct/"

df = pd.read_csv('TData9313_final4.csv', index_col=0)
df.index = pd.to_datetime(df.index)

for file in os.listdir(folder):
    try:
        ticker = file[:file.find(".")]
        a = df[ticker]
        f = pd.read_csv(folder + file, index_col=0, sep=";")
        f.index = pd.to_datetime(f.index, format="%b %d, %Y")
        print ticker
        for n, line in f.iterrows():
            if line['type'].strip() == "Dividend":
                scale = float(line['scale'])
                price = df[:n][ticker][-1]
                ratio = 1 - scale / price
                df[:n][ticker] *= ratio

            if line['type'].strip() == "Stock Split":
                rawscale = line['scale'].strip().split(":")
                splitScale = float(rawscale[1]) / float(rawscale[0])
                df[:n][ticker] *= splitScale

    except KeyError:
        print "Error:", ticker

    except IndexError:
        print "Index Error:", n

df.to_csv('TData9313_final5.csv')