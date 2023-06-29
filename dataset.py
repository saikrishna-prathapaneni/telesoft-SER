import os
import pandas as pd
import torchaudio
from torch.utils.data import DataLoader, Dataset
import torch


def speech_file_to_array_fn(path):
    speech_array, sampling_rate = torchaudio.load(path)
    resampler = torchaudio.transforms.Resample(sampling_rate, 8000)
    speech = resampler(speech_array).squeeze().numpy()
    return speech



class RecolaDataset(Dataset):
    def __init__(self, audio_paths, labels, tokenizer, max_length):
        self.audio_paths = audio_paths
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        audio_path = self.audio_paths[idx]
        label = self.labels[idx]

        waveform, sample_rate = torchaudio.load(audio_path)

        # Resample the waveform to a specific sample rate (e.g., 16000)
        resampled_waveform = torchaudio.transforms.Resample(sample_rate, 16000)(waveform)

        # Pad or truncate the waveform to the desired length (e.g., 16000 samples)
        waveform_length = resampled_waveform.size(1)
        if waveform_length < self.max_length:
            # Pad the waveform with zeros
            padded_waveform = torch.nn.functional.pad(resampled_waveform, (0, self.max_length - waveform_length))
        elif waveform_length > self.max_length:
            # Truncate the waveform
            padded_waveform = resampled_waveform[:, :self.max_length]
        else:
            padded_waveform = resampled_waveform
        #padded_waveform = padded_waveform.to(torch.float)
        inputs = self.tokenizer(
            padded_waveform.numpy()[0],
            padding='max_length',
            return_tensors="pt",
            max_length=self.max_length,
            sample_rate=400
        )
        print(inputs.shape)

        return {
            #'input_values':padded_waveform.squeeze(),
            'input_values': inputs['input_values'].squeeze(),
            'attention_mask':inputs['attention_mask'].squeeze(),
            'labels': torch.tensor(label-1,dtype= torch.float)
        }

 