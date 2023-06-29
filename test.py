import os
import torch
import torchaudio
import pandas as pd
import numpy as np
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import Wav2Vec2ForCTC, Wav2Vec2CTCTokenizer,Wav2Vec2Processor
from sklearn.model_selection import train_test_split
from dataset import RecolaDataset

Ravdess ="./archive/"

ravdess_directory_list = os.listdir(Ravdess)

file_emotion = []
file_path = []
for dir in ravdess_directory_list:
    # as their are 20 different actors in our previous directory we need to extract files for each actor.
    actor = os.listdir(Ravdess + dir)
    for file in actor:
        part = file.split('.')[0]
        part = part.split('-')
        # third part in each file represents the emotion associated to that file.
        file_emotion.append(int(part[2]))
        file_path.append(Ravdess + dir + '/' + file)
        
# dataframe for emotion of files
emotion_df = pd.DataFrame(file_emotion, columns=['Emotions'])
print(len(emotion_df))

# dataframe for path of files.
path_df = pd.DataFrame(file_path, columns=['Path'])
Ravdess_df = pd.concat([emotion_df, path_df], axis=1)


tokenizer = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h")
model.out_features=8
train_paths, val_paths, train_labels, val_labels = train_test_split(Ravdess_df['Path'].to_list(),Ravdess_df['Emotions'].to_list(),test_size=0.2, random_state=42)


train_dataset = RecolaDataset(train_paths, train_labels, tokenizer, 32000)



batch = train_dataset[0]
input_values = batch['input_values']


print(batch['input_values'].shape)
print(batch.shape)

#t= torch.randn(1,32000)
#print(model(t))