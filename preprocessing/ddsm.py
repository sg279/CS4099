import pandas as pd
import os
import pydicom as dicom
import cv2
from PIL import Image
import datetime
import numpy as np
from sklearn.utils import shuffle

def ddsm_cropped_image_conversion(mode):
    if mode == "val":
        data = pd.read_csv("../mass_case_description_test_set.csv").append(pd.read_csv("../calc_case_description_test_set.csv"))
    if mode == "test":
        data = pd.read_csv("../mass_case_description_train_set.csv").append(pd.read_csv("../calc_case_description_train_set.csv"))
        cutoff = round(data.shape[0]*0.2)
        print(data.shape[0])
        data = data[-cutoff:]
        print(data.shape[0])
    elif mode == "train":
        data = pd.read_csv("../mass_case_description_train_set.csv").append(pd.read_csv("../calc_case_description_train_set.csv"))
        print(data.shape[0])
        cutoff = round(data.shape[0] * 0.2)
        data = data[:-cutoff]
        print(data.shape[0])
    data["pathology"].replace("MALIGNANT", "M", inplace=True)
    data["pathology"].replace(["BENIGN","BENIGN_WITHOUT_CALLBACK" ], "B", inplace=True)
    m = data.loc[data["pathology"] == "M"]
    b = data.loc[data["pathology"] == "B"]
    path = "F:\\DDSM data\\cropped_pngs\\"
    source_path = "F:\DDSM data\CBIS-DDSM"
    os.makedirs(path + "\\" + mode + "\\M", exist_ok=True)
    os.makedirs(path + "\\" + mode + "\\B", exist_ok=True)
    for row in data.iterrows():
        image_folder = row[1]['cropped image file path'].split("/")[0]
        subfolder1 = os.listdir(os.path.join(source_path, image_folder))[0]
        subfolder2 = os.listdir(os.path.join(source_path, image_folder, subfolder1))[0]
        ds = dicom.dcmread(os.path.join(source_path, image_folder, subfolder1, subfolder2, "1-1.dcm"))
        pixel_array_numpy = ds.pixel_array
        image = image_folder+".png"
        cv2.imwrite(os.path.join(path,mode,row[1]["pathology"], image), pixel_array_numpy)

def ddsm_metadata():
    test = pd.read_csv("../mass_case_description_test_set.csv").append(pd.read_csv("../calc_case_description_test_set.csv"))
    data = pd.read_csv("../mass_case_description_train_set.csv").append(pd.read_csv("../calc_case_description_train_set.csv"))
    meta = pd.read_csv("../ddsm_metadata.csv")
    data["pathology"].replace("MALIGNANT", "M", inplace=True)
    data["pathology"].replace(["BENIGN","BENIGN_WITHOUT_CALLBACK" ], "B", inplace=True)
    m = data.loc[data["pathology"] == "M"]
    b = data.loc[data["pathology"] == "B"]
    path = "F:\\DDSM data\\pngs_balanced\\"
    source_path = "F:\DDSM data\CBIS-DDSM"

def ddsm_conversion(mode):
    if mode == "test":
        data = pd.read_csv("../mass_case_description_test_set.csv").append(pd.read_csv("../calc_case_description_test_set.csv"))
    else:
        data = pd.read_csv("../mass_case_description_train_set.csv").append(pd.read_csv("../calc_case_description_train_set.csv"))
        data=shuffle(data)
    data["pathology"].replace("MALIGNANT", "M", inplace=True)
    data["pathology"].replace(["BENIGN","BENIGN_WITHOUT_CALLBACK" ], "B", inplace=True)
    m = data.loc[data["pathology"] == "M"]
    b = data.loc[data["pathology"] == "B"]
    path = "F:\\DDSM data\\pngs_corrected\\"
    source_path = "F:\DDSM data\CBIS-DDSM"
    os.makedirs(path + "\\" + mode + "\\M", exist_ok=True)
    os.makedirs(path + "\\" + mode + "\\B", exist_ok=True)
    if mode == "test":
        for row in data.iterrows():
            image_folder = row[1]['image file path'].split("/")[0]
            subfolder1 = os.listdir(os.path.join(source_path, image_folder))[0]
            subfolder2 = os.listdir(os.path.join(source_path, image_folder, subfolder1))[0]
            ds = dicom.dcmread(os.path.join(source_path, image_folder, subfolder1, subfolder2, "1-1.dcm"))
            pixel_array_numpy = ds.pixel_array
            image = image_folder+".png"
            cv2.imwrite(os.path.join(path,mode,row[1]["pathology"], image), pixel_array_numpy)
    else:
        cutoff = round(data.shape[0] * 0.2)
        train_data = data[-cutoff:]
        val_data = data[:-cutoff]
        for row in train_data.iterrows():
            image_folder = row[1]['image file path'].split("/")[0]
            subfolder1 = os.listdir(os.path.join(source_path, image_folder))[0]
            subfolder2 = os.listdir(os.path.join(source_path, image_folder, subfolder1))[0]
            ds = dicom.dcmread(os.path.join(source_path, image_folder, subfolder1, subfolder2, "1-1.dcm"))
            pixel_array_numpy = ds.pixel_array
            image = image_folder+".png"
            cv2.imwrite(os.path.join(path,"train",row[1]["pathology"], image), pixel_array_numpy)
        for row in train_data.iterrows():
            image_folder = row[1]['image file path'].split("/")[0]
            subfolder1 = os.listdir(os.path.join(source_path, image_folder))[0]
            subfolder2 = os.listdir(os.path.join(source_path, image_folder, subfolder1))[0]
            ds = dicom.dcmread(os.path.join(source_path, image_folder, subfolder1, subfolder2, "1-1.dcm"))
            pixel_array_numpy = ds.pixel_array
            image = image_folder+".png"
            cv2.imwrite(os.path.join(path,"val",row[1]["pathology"], image), pixel_array_numpy)

def ddsm_test_conversion():
    data = pd.read_csv("../mass_case_description_test_set.csv").append(pd.read_csv("../calc_case_description_test_set.csv"))
    data["pathology"].replace("MALIGNANT", "M", inplace=True)
    data["pathology"].replace(["BENIGN","BENIGN_WITHOUT_CALLBACK" ], "B", inplace=True)
    path = "F:\\DDSM data\\pngs_corrected\\"
    source_path = "F:\DDSM data\CBIS-DDSM"
    os.makedirs(path + "\\" + "test" + "\\M", exist_ok=True)
    os.makedirs(path + "\\" + "test" + "\\B", exist_ok=True)
    for row in data.iterrows():
        image_folder = row[1]['image file path'].split("/")[0]
        subfolder1 = os.listdir(os.path.join(source_path, image_folder))[0]
        subfolder2 = os.listdir(os.path.join(source_path, image_folder, subfolder1))[0]
        ds = dicom.dcmread(os.path.join(source_path, image_folder, subfolder1, subfolder2, "1-1.dcm"))
        pixel_array_numpy = ds.pixel_array
        image = image_folder+".png"
        cv2.imwrite(os.path.join(path,"test",row[1]["pathology"], image), pixel_array_numpy)

def ddsm_train_conversion():
    mass_data = pd.read_csv("../mass_case_description_train_set.csv")
    calc_data = pd.read_csv("../calc_case_description_train_set.csv")
    mass_cutoff = round(mass_data.shape[0] * 0.2)
    calc_cutoff = round(calc_data.shape[0] * 0.2)
    train_mass_data = mass_data[:-mass_cutoff]
    val_mass_data = mass_data[-mass_cutoff:]
    train_calc_data = mass_data[:-calc_cutoff]
    val_calc_data = mass_data[-calc_cutoff:]
    train_data = train_mass_data.append(train_calc_data)
    val_data = val_mass_data.append(val_calc_data)
    train_data["pathology"].replace("MALIGNANT", "M", inplace=True)
    train_data["pathology"].replace(["BENIGN", "BENIGN_WITHOUT_CALLBACK"], "B", inplace=True)
    val_data["pathology"].replace("MALIGNANT", "M", inplace=True)
    val_data["pathology"].replace(["BENIGN", "BENIGN_WITHOUT_CALLBACK"], "B", inplace=True)
    path = "F:\\DDSM data\\pngs_corrected\\"
    source_path = "F:\DDSM data\CBIS-DDSM"
    os.makedirs(path + "\\" + "train" + "\\M", exist_ok=True)
    os.makedirs(path + "\\" + "train" + "\\B", exist_ok=True)
    os.makedirs(path + "\\" + "val" + "\\M", exist_ok=True)
    os.makedirs(path + "\\" + "val" + "\\B", exist_ok=True)
    for row in train_data.iterrows():
        image_folder = row[1]['image file path'].split("/")[0]
        subfolder1 = os.listdir(os.path.join(source_path, image_folder))[0]
        subfolder2 = os.listdir(os.path.join(source_path, image_folder, subfolder1))[0]
        ds = dicom.dcmread(os.path.join(source_path, image_folder, subfolder1, subfolder2, "1-1.dcm"))
        pixel_array_numpy = ds.pixel_array
        image = image_folder + ".png"
        cv2.imwrite(os.path.join(path, "train", row[1]["pathology"], image), pixel_array_numpy)
    for row in val_data.iterrows():
        image_folder = row[1]['image file path'].split("/")[0]
        subfolder1 = os.listdir(os.path.join(source_path, image_folder))[0]
        subfolder2 = os.listdir(os.path.join(source_path, image_folder, subfolder1))[0]
        ds = dicom.dcmread(os.path.join(source_path, image_folder, subfolder1, subfolder2, "1-1.dcm"))
        pixel_array_numpy = ds.pixel_array
        image = image_folder + ".png"
        cv2.imwrite(os.path.join(path, "val", row[1]["pathology"], image), pixel_array_numpy)

def time():
    data = pd.read_csv("../mass_case_description_test_set.csv").append(
        pd.read_csv("../calc_case_description_test_set.csv")).head(20)
    data["pathology"].replace("MALIGNANT", "M", inplace=True)
    data["pathology"].replace(["BENIGN", "BENIGN_WITHOUT_CALLBACK"], "B", inplace=True)
    path = "F:\\DDSM data\\pngs\\"
    source_path = "F:\DDSM data\CBIS-DDSM"
    start = datetime.datetime.now()
    for row in data.iterrows():
        image_folder = row[1]['image file path'].split("/")[0]
        subfolder1 = os.listdir(os.path.join(source_path, image_folder))[0]
        subfolder2 = os.listdir(os.path.join(source_path, image_folder, subfolder1))[0]
        ds = dicom.dcmread(os.path.join(source_path, image_folder, subfolder1, subfolder2, "1-1.dcm"))
        pixel_array_numpy = ds.pixel_array
    print(datetime.datetime.now()-start)
    start = datetime.datetime.now()
    for row in data.iterrows():
        image = row[1]['image file path'].split("/")[0]+".png"
        im = Image.open(os.path.join(path, "test", row[1]["pathology"], image))
        pixel_array_numpy = np.array(image)
    print(datetime.datetime.now()-start)

if __name__ == '__main__':
    # time()
    # ddsm_cropped_image_conversion("test")
    # ddsm_cropped_image_conversion("train")
    # ddsm_cropped_image_conversion("val")
    # ddsm_metadata()
    # ddsm_test_conversion()
    ddsm_train_conversion()