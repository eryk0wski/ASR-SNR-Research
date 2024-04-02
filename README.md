# ASR Polish Research Project

# SNR Research

## About Project

Project aims to research how noises impact speech recognition algorithms. Dataset used for the project was training batch from 'BIGOS v2 dataset', containing most of polish-speaking speech datasets. Models that were chosen for research:

- Wav2wec2 large tuned for polish by Jonatas Grosman
- Whisper large v3 

Noise datasets used for research:

- UrbanNoises8K
- Vehicle Interior Sounds Dataset 

Noises and speech audios were normalized to loudness of -20dB. Then mixed with different rates of SNR Ratios 100, 50, 25, 10, 5, 1, -1, -3, -10. Each SNR values was tested by each model and WER values were calculated as model quality rating. Other metric were also included: word with the most errors etc. Attempts have also been made to made some unique evaluation method using Chat GPT, but until now they were unsuccessful.


# Demo

Demo version of the project has been included as demo notebook. It is utilizing already prepared and randomly chosen recordings.

## Installation

Required libraries to install for project to work are listed in requirements file.
You can download them in bulk by running:

    $ pip install -r requirements.txt

 ## Configuration for your usage.
 Unzip VISC Noises dataset and UrbanNoises dataset and put them into data folder. Next steps are included in the notebooks.


## Testing times

For future testing I include my PC Setup and how much time each iteration has taken:
- CPU: Ryzen 7 5700X
- GPU: RTX 4060 Ti 16GB CUDA 12.4
- RAM 32 GB 3600 MHz

Whisper Large v3:
- One batch with 2500 sentences '130 min'. CUDA Included.
- One batch with 5000 sentences '280 min'. CUDA Included.
- One batch with 5000 sentences '930 min'. No CUDA.

Wav2wec2:
- One batch with 2500 sentences '250 min'. CUDA included.
- Three batches with 2500 sentences '689 min'. CUDA included.

# Model fine-tuning

I have also fine-tuned whisper-medium model to work better with speech mixed with vehicle interior noises.
Fine-tuning batch: 750 recordings + 150 evaluation. Speech and noise recordings normalized to -20dB. Recordings mixed with SNR value of 25 dB.
Model works, but available computational power didn't allow me to train it sufficiently.

## Fine tuned model usage

```python
# Specify the CUDA device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_id = "eryk7381/whisper-med-pol-car"
torch_dtype = torch.float16 # You can adjust the dtype if needed


# Load model and move it to CUDA
model = AutoModelForSpeechSeq2Seq.from_pretrained(
model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)


# Load processor
processor = AutoProcessor.from_pretrained(model_id)


# Create the pipeline with CUDA support
pipe = pipeline(
"automatic-speech-recognition",
model=model,
tokenizer=processor.tokenizer,
feature_extractor=processor.feature_extractor,
max_new_tokens=128,
chunk_length_s=30,
batch_size=16,
return_timestamps=True,
torch_dtype=torch_dtype,
device=device,
)

# Inputing your file path
audio_path = 'your_audio_path.wav'
sample = audio_path
result = pipe(sample, generate_kwargs={"language": "polish"})
print(result['text'])
```



## Training times
Training batches:
- 750 recordings - 22h
- 7500 recording - 202h (predicted)
- 75000 recordings - 2000h (predicted)

# References
Bigos v2 dataset: https://huggingface.co/datasets/michaljunczyk/pl-asr-bigos-v2

Wav2wec2 model: https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-polish

Whisper model: https://huggingface.co/openai/whisper-large-v3