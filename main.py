import os
import torch
import torchaudio
import pandas as pd
import numpy as np
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import Wav2Vec2ForCTC, Wav2Vec2CTCTokenizer,Wav2Vec2Processor
from model import EmotionModel
from sklearn.model_selection import train_test_split
from dataset import RecolaDataset
from torch import autocast
from torch.cuda.amp import GradScaler

Ravdess="./archive/"
max_length = 8000  # Max audio length in samples
batch_size = 1
learning_rate = 1e-4
num_epochs = 10




if __name__=="__main__":
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
    scaler = GradScaler()
    emotion_df = pd.DataFrame(file_emotion, columns=['Emotions'])
    print(len(emotion_df))

    # dataframe for path of files.
    path_df = pd.DataFrame(file_path, columns=['Path'])
    Ravdess_df = pd.concat([emotion_df, path_df], axis=1)

    # changing integers to actual emotions.
    #Ravdess_df.Emotions.replace({1:'neutral', 2:'calm', 3:'happy', 4:'sad', 5:'angry', 6:'fear', 7:'disgust', 8:'surprise'}, inplace=True)

    model_name ="facebook/wav2vec2-large-960h"

    Ravdess_df['Emotions'] = Ravdess_df['Emotions']-1
    tokenizer = Wav2Vec2Processor.from_pretrained(model_name,sampling_rate=16000)
    target_sampling_rate = tokenizer.feature_extractor.sampling_rate
    print(target_sampling_rate)
    model = EmotionModel.from_pretrained(model_name)
    model.out_features=8
    # labels = [0,1,2,3,4,5,6,7]
    # print(Ravdess_df[~Ravdess_df['Emotions'].isin(labels)])

    train_paths, val_paths, train_labels, val_labels = train_test_split(Ravdess_df['Path'].to_list(),Ravdess_df['Emotions'].to_list(),test_size=0.2, random_state=42)
    #print(Ravdess_df['Emotions'].to_list())
    # Prepare the datasets and data loaders
    #target_sampling_rate = modetokenizerl.sampling_rate

    train_dataset = RecolaDataset(train_paths, train_labels, tokenizer, max_length,)
    val_dataset = RecolaDataset(val_paths, val_labels, tokenizer, max_length)


    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Define optimizer and loss function
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.CrossEntropyLoss()

    # Fine-tuning loop
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_loss = 0
        total_correct = 0
        total_examples = 0
        for batch in train_loader:
            input_values = batch['input_values'].to(device)
            #attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
           

            with autocast(device_type='cuda',dtype=torch.float16):
                outputs = model(input_values=input_values)
                loss = loss_fn(outputs, labels)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            
            scaler.update()
            train_loss += loss.item()
            predicted_labels = torch.argmax(outputs, dim=1)
            correct = torch.sum(predicted_labels == labels).item()
            total_correct += correct
            total_examples += labels.size(0)
            print("train loss:", train_loss)

        #train_loss /= len(train_loader)
        train_loss /= len(train_loader)
        train_accuracy = total_correct / total_examples
        # Validation
        model.eval()
        val_loss = 0.0

        val_loss = 0
        total_correct = 0
        total_examples = 0
        with torch.no_grad():
                for batch in val_loader:
                    input_values = batch['input_values'].to(device)
                    labels = batch['labels'].to(device)

                    outputs = model(input_values=input_values)
                    loss = loss_fn(outputs, labels)


                    predicted_labels = torch.argmax(outputs, dim=1)
                    correct = torch.sum(predicted_labels == labels).item()
                    total_correct += correct
                    total_examples += labels.size(0)

                    val_loss += loss.item()

                val_loss /= len(val_loader)
                val_accuracy = total_correct / total_examples

        print(f'Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Train Accuracy: {train_accuracy:.4f} - Val Accuracy: {val_accuracy:.4f}')
        checkpoint_path = os.path.join("./", f'epoch_{epoch+1}_{val_loss}_{val_accuracy}.pt')
        torch.save(model.state_dict(), checkpoint_path)
