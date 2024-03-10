import numpy as np
import librosa
import soundfile

'''
def add_noise_snr(audio, noise, snr):
    clean_db = 10 * np.log10(np.mean(audio ** 2) + 1e-4)
    audio_length = audio.shape[0]
    if noise.shape[0] <= audio_length:
        shortage = audio_length - noise.shape[0]
        noise = np.pad(noise, (0, shortage), 'wrap')
    noise_db = 10 * np.log10(np.mean(noise ** 2) + 1e-4)
    add_noise = np.sqrt(10 ** ((clean_db - noise_db - snr) / 10)) * noise
    mix_audio = audio + add_noise
    return mix_audio
    '''

def add_noise_snr(audio, noise, snr):
    clean_db = 10 * np.log10(np.mean(audio ** 2) + 1e-4)
    audio_length = audio.shape[0]

    if noise.shape[0] < audio_length:
        shortage = audio_length - noise.shape[0]
        noise = np.pad(noise, (0, shortage), 'wrap')
    elif noise.shape[0] > audio_length:
        noise = noise[:audio_length]

    noise_db = 10 * np.log10(np.mean(noise ** 2) + 1e-4)
    add_noise = np.sqrt(10 ** ((clean_db - noise_db - snr) / 10)) * noise
    mix_audio = audio + add_noise
    return mix_audio

def read_audio(file_audio):
    audio, sr = librosa.load(file_audio, sr=8000, mono=True)
    return audio
def save_wav(audio, fx, sr = 8000):
    soundfile.write(fx, audio, sr, "PCM_16")

def mixer(signal_file_path, noise_file_path, SNR, save_path):
    audio = read_audio(signal_file_path)
    noise = read_audio(noise_file_path)
    mixed_audio = add_noise_snr(audio,noise,SNR)
    save_wav(mixed_audio, save_path)
    


#wav_file = ".\commons_pl\clips\common_voice_pl_20547889.mp3"
#audio = read_audio(wav_file)
#noise_file = r".\VISC_Dataset_SON\6_35.wav"
#noise = read_audio(noise_file)

#audio_noise = add_noise_snr(audio, noise, snr = 0)
#save_wav(audio_noise, "./mixed_sounds/mixed_commons/audio_noise_0.wav")

#audio_noise = add_noise_snr(audio, noise, snr = 5)
#save_wav(audio_noise, "./mixed_sounds/mixed_commons/audio_noise_5.wav")

#audio_noise = add_noise_snr(audio, noise, snr = 10)
#save_wav(audio_noise, "./mixed_sounds/mixed_commons/audio_noise_10.wav")