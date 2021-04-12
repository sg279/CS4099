import pandas as pd
import os
import pydicom as dicom
import cv2
from PIL import Image
import datetime
import numpy as np
from sklearn.utils import shuffle
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler


"""
Remove and rename columns from test metadata, combining mass and calcification data
"""
def ddsm_test_metadata_conversion():
    # Load mass and calc CSVs
    data = pd.read_csv("../mass_case_description_test_set.csv").append(pd.read_csv("../calc_case_description_test_set.csv"))
    # Replace pathologies to create two classes
    data["pathology"].replace("MALIGNANT", "M", inplace=True)
    data["pathology"].replace(["BENIGN","BENIGN_WITHOUT_CALLBACK" ], "B", inplace=True)
    # Create dataframes for benign and malignant cases
    test_b_metadata = pd.DataFrame(
        columns=['pathology','label', 'density', 'side', 'view', 'abnormality_id', 'abnormality_type', 'shape', 'margins',
                 'assessment', 'subtlety', 'calc_distribution', 'calc_type'])
    test_m_metadata = pd.DataFrame(
        columns=['pathology','label', 'density', 'side', 'view', 'abnormality_id', 'abnormality_type', 'shape', 'margins',
                 'assessment', 'subtlety', 'calc_distribution', 'calc_type'])
    i=0
    # For each row, add metadata to dataframes
    for row in data.iterrows():
        i = i + 1
        pathology = row[1]['pathology']
        label = row[1]['image file path'].split("/")[0]
        density = row[1]['breast_density']
        side = row[1]['left or right breast']
        view = row[1]['image view']
        abnormality_id = row[1]['abnormality id']
        abnormality_type = row[1]['abnormality type']
        shape = row[1]['mass shape']
        margins = row[1]['mass margins']
        assessment = row[1]['assessment']
        subtlety = row[1]['subtlety']
        calc_distribution = row[1]['calc distribution']
        calc_type = row[1]['calc type']
        if row[1]["pathology"] == 'M':
            test_m_metadata.loc[label] = [pathology,label, density, side, view, abnormality_id, abnormality_type, shape, margins,
                                          assessment, subtlety, calc_distribution, calc_type]
        else:
            test_b_metadata.loc[label] = [pathology,label, density, side, view, abnormality_id, abnormality_type, shape, margins,
                                          assessment, subtlety, calc_distribution, calc_type]
    test_b_metadata.append(test_m_metadata).to_csv("./unprocessed_test_metadata.csv")


"""
Remove and rename columns from training metadata, combining mass and calcification data 
and splitting into train and validation
"""
def ddsm_train_metadata_conversion():
    mass_data = pd.read_csv("../mass_case_description_train_set.csv")
    calc_data = pd.read_csv("../calc_case_description_train_set.csv")
    # Get the validation cutoffs for masses and calcifications and split dataframes
    mass_cutoff = round(mass_data.shape[0] * 0.2)
    calc_cutoff = round(calc_data.shape[0] * 0.2)
    train_mass_data = mass_data[:-mass_cutoff]
    val_mass_data = mass_data[-mass_cutoff:]
    print(val_mass_data.shape)
    exit()
    train_calc_data = calc_data[:-calc_cutoff]
    val_calc_data = calc_data[-calc_cutoff:]
    # Join mass and calcification dataframes
    train_data = train_mass_data.append(train_calc_data, ignore_index=True)
    val_data = val_mass_data.append(val_calc_data, ignore_index=True)
    # Replace pathologies to create two classes
    train_data["pathology"].replace("MALIGNANT", "M", inplace=True)
    train_data["pathology"].replace(["BENIGN", "BENIGN_WITHOUT_CALLBACK"], "B", inplace=True)
    val_data["pathology"].replace("MALIGNANT", "M", inplace=True)
    val_data["pathology"].replace(["BENIGN", "BENIGN_WITHOUT_CALLBACK"], "B", inplace=True)
    i = 0
    # Create dataframes for benign and malignant training and validation cases
    training_b_metadata = pd.DataFrame(columns=['pathology','label', 'density', 'side', 'view', 'abnormality_id', 'abnormality_type','shape', 'margins', 'assessment','subtlety', 'calc_distribution', 'calc_type'])
    training_m_metadata = pd.DataFrame(columns=['pathology','label', 'density', 'side', 'view', 'abnormality_id', 'abnormality_type','shape', 'margins', 'assessment','subtlety', 'calc_distribution', 'calc_type'])
    val_b_metadata = pd.DataFrame(columns=['pathology','label', 'density', 'side', 'view', 'abnormality_id', 'abnormality_type','shape', 'margins', 'assessment','subtlety', 'calc_distribution', 'calc_type'])
    val_m_metadata = pd.DataFrame(columns=['pathology','label', 'density', 'side', 'view', 'abnormality_id', 'abnormality_type','shape', 'margins', 'assessment','subtlety', 'calc_distribution', 'calc_type'])
    # For each row, add metadata to dataframes
    for row in train_data.iterrows():
        i=i+1
        label = str(i)
        pathology = row[1]['pathology']
        density = row[1]['breast_density']
        side = row[1]['left or right breast']
        view = row[1]['image view']
        abnormality_id = row[1]['abnormality id']
        abnormality_type = row[1]['abnormality type']
        shape = row[1]['mass shape']
        margins = row[1]['mass margins']
        assessment = row[1]['assessment']
        subtlety = row[1]['subtlety']
        calc_distribution = row[1]['calc distribution']
        calc_type = row[1]['calc type']
        if row[1]["pathology"]=='M':
            training_m_metadata.loc[i]=[pathology,label,density,side,view,abnormality_id,abnormality_type,shape,margins,assessment,subtlety, calc_distribution, calc_type]
        elif row[1]["pathology"]=='B':
            training_b_metadata.loc[i]=[pathology,label,density,side,view,abnormality_id,abnormality_type,shape,margins,assessment,subtlety, calc_distribution, calc_type]
    for row in val_data.iterrows():
        i = i + 1
        label = str(i)
        pathology = row[1]['pathology']
        density = row[1]['breast_density']
        side = row[1]['left or right breast']
        view = row[1]['image view']
        abnormality_id = row[1]['abnormality id']
        abnormality_type = row[1]['abnormality type']
        shape = row[1]['mass shape']
        margins = row[1]['mass margins']
        assessment = row[1]['assessment']
        subtlety = row[1]['subtlety']
        calc_distribution = row[1]['calc distribution']
        calc_type = row[1]['calc type']
        if row[1]["pathology"] == 'M':
            val_m_metadata.loc[i] = [pathology,label, density, side, view, abnormality_id, abnormality_type, shape, margins,
                                          assessment, subtlety, calc_distribution, calc_type]
        else:
            val_b_metadata.loc[i] = [pathology,label, density, side, view, abnormality_id, abnormality_type, shape, margins,
                                          assessment, subtlety, calc_distribution, calc_type]
    training_b_metadata.append(training_m_metadata).to_csv("./unprocessed_training_metadata.csv")
    val_b_metadata.append(val_m_metadata).to_csv("./unprocessed_val_metadata.csv")

"""
Get the absolute file paths for all files in a directory
"""
def absoluteFilePaths(directory):
    files = []
    for dirpath, _, filenames in os.walk(directory):
        for f in filenames:
            files.append(os.path.abspath(os.path.join(dirpath, f)))
    return files

"""
Pre process metadata to be used for classification
"""
def process_radiologist_metadata():
    training = pd.read_csv("./unprocessed_training_calc_metadata.csv")
    val = pd.read_csv("./unprocessed_val_calc_metadata.csv")
    test = pd.read_csv("./unprocessed_test_calc_metadata.csv")
    # Create dataframe of all data used for getting categorical encodings
    scaler_df = training.append(val.append(test))
    scaler_df.drop(['Unnamed: 0','abnormality_id', 'pathology'], axis=1, inplace=True)
    training_rows = training.shape[0]
    val_rows = val.shape[0]
    bin_cols = ['side', 'view', 'abnormality_type']
    cat_columns = ['calc_distribution', 'calc_type']
    num_columns = ['assessment', 'subtlety']
    minMax = MinMaxScaler()
    encoded = None
    # Get one hot encodings and drop columns
    for c in cat_columns:
        ohe = pd.get_dummies(scaler_df[c], dummy_na=True)
        if ohe[np.nan].values.max() == 0:
            ohe.drop(np.nan, axis=1, inplace=True)
        encoded = pd.concat([encoded, ohe], axis=1)
    for c in cat_columns:
        scaler_df = scaler_df.drop(c, axis=1)
    # Get binary encodings
    for c in bin_cols:
        scaler_df[c]=scaler_df[c].astype('category').cat.codes
    scaler_df = pd.concat([encoded, scaler_df], axis=1)
    # Fill nan values in numerical columns
    for c in num_columns:
        scaler_df[c] = scaler_df[c].fillna(0)
    # Split scaler dataframe back intro training, validation, and test
    training = scaler_df[:training_rows]
    val = scaler_df[training_rows:training_rows+val_rows]
    test = scaler_df[training_rows+val_rows:]
    # Create dataframe of training and validation data for getting min max values to use for scaling
    scaler_df = scaler_df[:training_rows+val_rows]
    minMax.fit(scaler_df[num_columns])
    # Scale numeric values
    training[num_columns]=minMax.transform(training[num_columns])
    val[num_columns] = minMax.transform(val[num_columns])
    test[num_columns] = minMax.transform(test[num_columns])
    print(training.shape[1])
    training.to_csv("./all_calc_training_metadata.csv", index=False)
    val.to_csv("./all_calc_val_metadata.csv", index=False)
    test.to_csv("./all_calc_test_metadata.csv", index=False)

"""
Identical to previous method but drops different columns
"""
def process_metadata():
    training = pd.read_csv("./unprocessed_training_metadata.csv")
    val = pd.read_csv("./unprocessed_val_metadata.csv")
    test = pd.read_csv("./unprocessed_test_metadata.csv")
    scaler_df = training.append(val.append(test))
    scaler_df.drop(['Unnamed: 0','abnormality_id', 'pathology','shape', 'margins', 'calc_distribution','calc_type','assessment', 'subtlety'], axis=1, inplace=True)
    training_rows = training.shape[0]
    val_rows = val.shape[0]
    bin_cols = ['side', 'view', 'abnormality_type']
    cat_columns = []
    num_columns = ['density']
    minMax = MinMaxScaler()
    encoded = None
    for c in cat_columns:
        ohe = pd.get_dummies(scaler_df[c], dummy_na=True)
        if ohe[np.nan].values.max() == 0:
            ohe.drop(np.nan, axis=1, inplace=True)
        encoded = pd.concat([encoded, ohe], axis=1)
    for c in cat_columns:
        scaler_df = scaler_df.drop(c, axis=1)
    for c in bin_cols:
        scaler_df[c]=scaler_df[c].astype('category').cat.codes
    scaler_df = pd.concat([encoded, scaler_df], axis=1)
    for c in num_columns:
        scaler_df[c] = scaler_df[c].fillna(0)
    training = scaler_df[:training_rows]
    val = scaler_df[training_rows:training_rows+val_rows]
    test = scaler_df[training_rows+val_rows:]
    scaler_df = scaler_df[:training_rows+val_rows]
    minMax.fit(scaler_df[num_columns])
    training[num_columns]=minMax.transform(training[num_columns])
    val[num_columns] = minMax.transform(val[num_columns])
    test[num_columns] = minMax.transform(test[num_columns])
    print(training.shape[1])
    training.to_csv("./no_radiologist_training_metadata.csv", index=False)
    val.to_csv("./no_radiologist_val_metadata.csv", index=False)
    test.to_csv("./no_radiologist_test_metadata.csv", index=False)

"""
Convert CBIS DDSM DICOM test data images to PNGs in malignant and benign directories
"""
def ddsm_test_conversion():
    # Load mass and calc CSVs
    data = pd.read_csv("../mass_case_description_test_set.csv").append(pd.read_csv("../calc_case_description_test_set.csv"))
    # Replace pathologies to create two classes
    data["pathology"].replace("MALIGNANT", "M", inplace=True)
    data["pathology"].replace(["BENIGN","BENIGN_WITHOUT_CALLBACK" ], "B", inplace=True)
    path = "F:\\DDSM data\\pngs\\"
    source_path = "F:\DDSM data\CBIS-DDSM"
    os.makedirs(path + "\\" + "test" + "\\M", exist_ok=True)
    os.makedirs(path + "\\" + "test" + "\\B", exist_ok=True)
    # For each row
    for row in data.iterrows():
        # Get the path of the source image
        image_folder = row[1]['image file path'].split("/")[0]
        subfolder1 = os.listdir(os.path.join(source_path, image_folder))[0]
        subfolder2 = os.listdir(os.path.join(source_path, image_folder, subfolder1))[0]
        # Read the DICOM into a pixel array
        ds = dicom.dcmread(os.path.join(source_path, image_folder, subfolder1, subfolder2, "1-1.dcm"))
        pixel_array_numpy = ds.pixel_array
        image = image_folder+".png"
        # Write the pixel array to disk in a directory for the pathology
        cv2.imwrite(os.path.join(path,"test",row[1]["pathology"], image), pixel_array_numpy)

"""
Convert CBIS DDSM DICOM training data images to PNGs in malignant and benign training and validation directories
"""
def ddsm_train_conversion():
    mass_data = pd.read_csv("../mass_case_description_train_set.csv")
    calc_data = pd.read_csv("../calc_case_description_train_set.csv")
    # Get the validation cutoffs for masses and calcifications and split dataframes
    mass_cutoff = round(mass_data.shape[0] * 0.2)
    calc_cutoff = round(calc_data.shape[0] * 0.2)
    train_mass_data = mass_data[:-mass_cutoff]
    val_mass_data = mass_data[-mass_cutoff:]
    train_calc_data = calc_data[:-calc_cutoff]
    val_calc_data = calc_data[-calc_cutoff:]
    # Join mass and calcification datframes
    train_data = train_mass_data.append(train_calc_data, ignore_index=True)
    val_data = val_mass_data.append(val_calc_data, ignore_index=True)
    # Replace pathologies to create two classes
    train_data["pathology"].replace("MALIGNANT", "M", inplace=True)
    train_data["pathology"].replace(["BENIGN", "BENIGN_WITHOUT_CALLBACK"], "B", inplace=True)
    val_data["pathology"].replace("MALIGNANT", "M", inplace=True)
    val_data["pathology"].replace(["BENIGN", "BENIGN_WITHOUT_CALLBACK"], "B", inplace=True)
    path = "F:\\DDSM data\\pngs\\"
    source_path = "F:\DDSM data\CBIS-DDSM"
    os.makedirs(path + "\\" + "train" + "\\M", exist_ok=True)
    os.makedirs(path + "\\" + "train" + "\\B", exist_ok=True)
    os.makedirs(path + "\\" + "val" + "\\M", exist_ok=True)
    os.makedirs(path + "\\" + "val" + "\\B", exist_ok=True)
    i=0
    # Same process as for test conversion but done for training and validation data
    for row in train_data.iterrows():
        i=i+1
        image_folder = row[1]['image file path'].split("/")[0]
        subfolder1 = os.listdir(os.path.join(source_path, image_folder))[0]
        subfolder2 = os.listdir(os.path.join(source_path, image_folder, subfolder1))[0]
        ds = dicom.dcmread(os.path.join(source_path, image_folder, subfolder1, subfolder2, "1-1.dcm"))
        pixel_array_numpy = ds.pixel_array
        image = str(i) + ".png"
        cv2.imwrite(os.path.join(path, "train", row[1]["pathology"], image), pixel_array_numpy)
    for row in val_data.iterrows():
        i = i + 1
        image_folder = row[1]['image file path'].split("/")[0]
        subfolder1 = os.listdir(os.path.join(source_path, image_folder))[0]
        subfolder2 = os.listdir(os.path.join(source_path, image_folder, subfolder1))[0]
        ds = dicom.dcmread(os.path.join(source_path, image_folder, subfolder1, subfolder2, "1-1.dcm"))
        pixel_array_numpy = ds.pixel_array
        image = str(i) + ".png"
        cv2.imwrite(os.path.join(path, "val", row[1]["pathology"], image), pixel_array_numpy)


if __name__ == '__main__':
    # time()
    # ddsm_cropped_image_conversion("test")
    # ddsm_cropped_image_conversion("train")
    # ddsm_cropped_image_conversion("val")
    # ddsm_metadata()
    # ddsm_train_conversion()
    # ddsm_train_metadata_conversion()
    # np.load("./models/classes/test_classes")
    # ddsm_test_metadata_conversion()
    # process_metadata()
    process_radiologist_metadata()
    # ddsm_train_metadata_conversion()