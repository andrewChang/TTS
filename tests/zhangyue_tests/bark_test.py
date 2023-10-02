from transformers import AutoProcessor, BarkModel
from transformers import AutoTokenizer
import scipy


# processor = AutoProcessor.from_pretrained("suno/bark")
# tokenizer = AutoTokenizer.from_pretrained("suno/bark")
# model = BarkModel.from_pretrained("suno/bark")

# processor.save_pretrained("/root/autodl-tmp/model/my_bark")
# tokenizer.save_pretrained("/root/autodl-tmp/model/my_bark")
# model.save_pretrained("/root/autodl-tmp/model/my_bark")

processor = AutoProcessor.from_pretrained("/root/autodl-tmp/model/my_bark")
tokenizer = AutoTokenizer.from_pretrained("/root/autodl-tmp/model/my_bark")
model = BarkModel.from_pretrained("/root/autodl-tmp/model/my_bark")

voice_preset = "v2/zh_speaker_8""
inputs = processor("[sighs]自相矛盾的成语故事讲的是一个在集市上卖矛和卖盾的故事♪", voice_preset=voice_preset)
audio_array = model.generate(**inputs,pad_token_id=tokenizer.eos_token_id)
audio_array = audio_array.cpu().numpy().squeeze()
sample_rate = model.generation_config.sample_rate
Audio(audio_array, rate=sample_rate)
print(audio_array)
print(f'---Finish generating----')
scipy.io.wavfile.write("bark_out.wav", rate=sample_rate, data=audio_array)
