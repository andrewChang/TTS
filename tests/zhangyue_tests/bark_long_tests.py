from transformers import AutoProcessor, BarkModel
from transformers import AutoTokenizer
import scipy
import nltk  # we'll use this to split into sentences
import numpy as np



processor = AutoProcessor.from_pretrained("/root/autodl-tmp/model/my_bark")
tokenizer = AutoTokenizer.from_pretrained("/root/autodl-tmp/model/my_bark")
model = BarkModel.from_pretrained("/root/autodl-tmp/model/my_bark")
sample_rate = model.generation_config.sample_rate
nltk.download('punkt')

script = """
Hey, have you heard about this new text-to-audio model called "Bark"? 
Apparently, it's the most realistic and natural-sounding text-to-audio model 
out there right now.
""".replace("\n", " ").strip()

sentences = nltk.sent_tokenize(script)
print(f'------sentences.size()={len(sentences)}------')
print(f'------sentences={sentences}------')

SPEAKER = "v2/en_speaker_6"
silence = np.zeros(int(0.25 * sample_rate))  # quarter second of silence

pieces = []
for sentence in sentences:
    inputs = processor(sentence, voice_preset=SPEAKER)
    audio_array = model.generate(**inputs,pad_token_id=tokenizer.eos_token_id)
    pieces += [audio_array, silence.copy()]

print(pieces)
final_audio_array = np.concatenate(pieces)
# final_audio_array = pieces
scipy.io.wavfile.write("bark_out.wav", rate=sample_rate, data=final_audio_array)

# Audio(final_audio_array, rate=sample_rate)