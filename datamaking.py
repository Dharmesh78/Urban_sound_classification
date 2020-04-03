# Load various imports
import numpy as np
import pandas as pd
import os
import librosa

def extract_features(file_name):
    try:
        audio, sample_rate = librosa.load(file_name)
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfccsscaled = np.mean(mfccs.T, axis=0)

    except Exception as e:
        print("Error encountered while parsing file: ", file_name)
        return None

    return mfccsscaled




# Set the path to the full UrbanSound dataset
fulldatasetpath = 'UrbanSound8K/audio/'

metadata = pd.read_csv('UrbanSound8K/metadata/UrbanSound8K.csv')

features = []

# Iterate through each sound file and extract the features
i=0
for index, row in metadata.iterrows():
    print("%d / %d"%(i,metadata.shape[0]))
    i+=1
    file_name = os.path.join(fulldatasetpath, 'fold' + str(row["fold"]) + '/',
                             str(row["slice_file_name"]))

    class_label = row["class"]
    data = extract_features(file_name)

    features.append([data, class_label])

# Convert into a Panda dataframe
df = pd.DataFrame(features, columns=['feature', 'class_label'])

print('Finished feature extraction from ', len(df), ' files')

df.to_csv('audioData_mlp.csv', index=False,index_label = True, encoding='utf-8')