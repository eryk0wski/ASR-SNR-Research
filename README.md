# ASR Polish Research Project

## About Project

Project aims to research how noises impact speech recognition algorithms. Dataset used for the project was training batch from 'BIGOS v2 dataset', containing most of polish-speaking speech datasets. Models that were chosen for research:

- Wav2wec2 large tuned for polish by Jonatas Grosman
- Whisper large v3 

Noise datasets used for research:

- UrbanNoises8K
- Vehicle Interior Sounds Dataset 

Noises and speech audios were normalized to loudness of -20dB. Then mixed with different rates of SNR Ratios 100,50,25,10,5,1,-1,-3,-10. Each SNR values was tested by each model and WER values were calculated as model quality rating.


## Installation

Required libraries to install for project to work are listed in requirements file.
You can download them in bulk by running:

    $ pip install -r requirements.txt

Otherwise you can install them one by one, as listed below:
    pip install datasets==2.18.0
    pip install librosa==0.10.1
    pip install numpy==1.24.3
    pip install pandas==2.2.1
    pip install pytest==8.1.1
    pip install soundfile==0.12.1




 ## Configuration for your usage.
 Unzip VISC Noises dataset and UrbanNoises dataset and put them into data folder.



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

## Training times
For future changes .....


## References
Bigos dataset:
Wav2wec2 model:
Whisper model: