import pandas as pd
import os
from PIL import Image

def mias_conversion(mode):

    data = pd.read_csv("mias.csv")
    n = data.loc[data["severity"] == "N"]
    m = data.loc[data["severity"]=="M"]
    b = data.loc[data["severity"]=="B"]
    if  mode == "val":
        data = n[-round(0.2*len(n)):].append(m[-round(0.2*len(m)):]).append(b[-round(0.2*len(b)):])
    elif mode == "test":
        data = n[-round(0.4*len(n)):-round(0.2*len(n))].append(m[-round(0.4*len(m)):-round(0.2*len(m))]).append(b[-round(0.4*len(b)):-round(0.2*len(b))])
    elif mode == "train":
        data = n[:-round(0.4*len(n))].append(m[:-round(0.4*len(m))]).append(b[:-round(0.4*len(b))])
    path = ".\mias\\"

    os.makedirs(path +"\\"+mode+"\\N", exist_ok=True)
    os.makedirs(path +"\\"+mode+"\\B", exist_ok=True)
    os.makedirs(path +"\\"+mode+"\\M", exist_ok=True)
    for row in data.iterrows():
        Image.open(path + row[1]["reference_number"] + ".pgm").save(
            os.path.join(path,mode, row[1]["severity"], row[1]["reference_number"] + '.png'))