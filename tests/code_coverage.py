import os

os.system('python3 bin/audioldm2 -t \"A toilet flushing and water trickling\"')
os.system('python3 bin/audioldm2 -t \"A toilet flushing and water trickling\" --model_name audioldm_48k')
os.system('python3 bin/audioldm2 -t \"A toilet flushing and water trickling\" --model_name audioldm_16k_crossattn_t5')
os.system('python3 bin/audioldm2 -tl /mnt/bn/lqhaoheliu/project/AudioLDM2/batch.lst')
os.system('python3 bin/audioldm2 -tl /mnt/bn/lqhaoheliu/project/AudioLDM2/batch.lst --model_name audioldm_48k')
os.system('python3 bin/audioldm2 -t \"A female reporter is speaking full of emotion\" --transcription \"wish you have a good day\"')