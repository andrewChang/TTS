import unittest
import os
import tempfile
from TTS.api import TTS


class TTSTalker():
    def __init__(self) -> None:
        model_name_list = TTS().list_models()
        #model_name = TTS("/root/.local/share/tts/tts_models--multilingual--multi-dataset--xtts_v1", gpu=True).list_models()[0]
        print(f'-------TTSTalker.models={model_name_list}--------')
        model_name = model_name_list[0]
        self.tts = TTS(model_name)

    def test(self, text, language='en'):

        tempf  = tempfile.NamedTemporaryFile(
                delete = False,
                suffix = ('.'+'wav'),
            )
        print(f'-------speaker={self.tts.speakers[0]},language={language},file_path={tempf.name}--------')
        self.tts.tts_to_file(text, speaker=self.tts.speakers[0], language=language, file_path=tempf.name)

        return tempf.name
    
if __name__=="__main__":
    tts = TTSTalker()
    tts.test(text= " hello ,nice to meet you ")


# class TestApi(unittest.TestCase):
#     def test_create(self):
#         model_name = TTS().list_models()[0]
#         print(f'-------TTSTalker.model_name={model_name}--------')





