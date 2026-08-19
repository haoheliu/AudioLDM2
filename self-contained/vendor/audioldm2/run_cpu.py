from diffusers import AudioLDM2Pipeline
import torch
import scipy

repo_id = "cvssp/audioldm2"
pipe = AudioLDM2Pipeline.from_pretrained(repo_id, torch_dtype=torch.float)
pipe = pipe.to("cpu")

#text prompt 
prompt = "Techno music with a strong, upbeat tempo and high melodic riffs."
audio = pipe(prompt, num_inference_steps=200, audio_length_in_s=15.0).audios[0]
# audio_length_in_s = 생성 음악 시간.

audiofile_name = "techno.wav"
scipy.io.wavfile.write(audiofile_name, rate=16000, data=audio)