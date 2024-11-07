import sys
import os.path as osp

import safetensors.torch
now_dir = osp.dirname(osp.abspath(__file__))
sys.path.append(now_dir)
comfy_utils = sys.modules["utils"]
import time
import os
import shutil
import torch
import numpy as np
import tempfile
import torchaudio
import folder_paths
import safetensors
from tqdm import tqdm
import py3langid as langid
from transformers import pipeline
from . import utils
sys.modules["utils"] = utils
from utils.util import load_config
from pydub import AudioSegment,silence
from huggingface_hub import hf_hub_download,snapshot_download
from models.tts.maskgct.maskgct_utils import *
aifsh_models = osp.join(folder_paths.models_dir, "AIFSH")
maskgct_dir = osp.join(aifsh_models,"MaskGCT")
openai_dir = osp.join(aifsh_models,"whisper-large-v3-turbo")
wav2vec_dir = osp.join(aifsh_models,"w2v-bert-2.0")
tmp_dir = osp.join(now_dir, "tmp")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sys.modules["utils"] = comfy_utils

def load_models():
    cfg_path = osp.join(now_dir,"models/tts/maskgct/config/maskgct.json")
    cfg = load_config(cfg_path)
    # 1. build semantic model (w2v-bert-2.0)
    semantic_model, semantic_mean, semantic_std = build_semantic_model(device,wav2vec_dir)
    # 2. build semantic codec
    semantic_codec = build_semantic_codec(cfg.model.semantic_codec, device)
    # 3. build acoustic codec
    codec_encoder, codec_decoder = build_acoustic_codec(cfg.model.acoustic_codec, device)
    # 4. build t2s model
    t2s_model = build_t2s_model(cfg.model.t2s_model, device)
    # 5. build s2a model
    s2a_model_1layer = build_s2a_model(cfg.model.s2a_model.s2a_1layer, device)
    s2a_model_full =  build_s2a_model(cfg.model.s2a_model.s2a_full, device)

    # load semantic codec
    safetensors.torch.load_model(semantic_codec, osp.join(maskgct_dir,"semantic_codec/model.safetensors"))
    # load acoustic codec
    safetensors.torch.load_model(codec_encoder, osp.join(maskgct_dir,"acoustic_codec/model.safetensors"))
    safetensors.torch.load_model(codec_decoder, osp.join(maskgct_dir,"acoustic_codec/model_1.safetensors"))
    # load t2s model
    safetensors.torch.load_model(t2s_model, osp.join(maskgct_dir,"t2s_model/model.safetensors"))
    # load s2a model
    safetensors.torch.load_model(s2a_model_1layer,osp.join(maskgct_dir,"s2a_model/s2a_model_1layer/model.safetensors"))
    safetensors.torch.load_model(s2a_model_full, osp.join(maskgct_dir,"s2a_model/s2a_model_full/model.safetensors"))

    maskgct_inference_pipeline = MaskGCT_Inference_Pipeline(
        semantic_model,
        semantic_codec,
        codec_encoder,
        codec_decoder,
        t2s_model,
        s2a_model_1layer,
        s2a_model_full,
        semantic_mean,
        semantic_std,
        device,
        wav2vec_dir
    )
    return maskgct_inference_pipeline
    
class MaskGCTNode:
    def __init__(self):
        if not osp.exists(osp.join(maskgct_dir,"s2a_model/s2a_model_full/model.safetensors")):
            # download semantic codec ckpt
            snapshot_download("amphion/MaskGCT",local_dir=maskgct_dir)
        
        if not osp.exists(osp.join(openai_dir,"model.safetensors")):
            snapshot_download(repo_id="openai/whisper-large-v3-turbo",
                              local_dir=openai_dir)
        
        if not osp.exists(osp.join(wav2vec_dir,"model.safetensors")):
            snapshot_download(repo_id="facebook/w2v-bert-2.0",local_dir=wav2vec_dir,
                              ignore_patterns=["*.pt"])
        self.pipe = None
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":{
                "target_text":("TEXT",),
                "prompt_wav":("AUDIO",),
                "store_in_varm":("BOOLEAN",{
                    "default":False
                }),
                "seed":("INT",{
                    "default": 42,
                })
            },
            "optional":{
                "prompt_text":("TEXT",),
            }
        }
    
    RETURN_TYPES = ("AUDIO",)
    #RETURN_NAMES = ("image_output_name",)

    FUNCTION = "gen_audio"

    #OUTPUT_NODE = False

    CATEGORY = "AIFSH_MaskGCT"

    def gen_audio(self,target_text,prompt_wav,store_in_varm,seed,prompt_text=None):
        # build model
        torch.manual_seed(seed)
        if self.pipe is None:
            self.pipe = load_models()
        # inference
        print("Converting audio...")
        os.makedirs(tmp_dir,exist_ok=True)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav",dir=tmp_dir) as f:
            ref_audio_orig = osp.join(tmp_dir,"tmp_ref_audio.wav")
            waveform = prompt_wav["waveform"].squeeze(0)

            torchaudio.save(ref_audio_orig,waveform,prompt_wav["sample_rate"])
            aseg = AudioSegment.from_file(ref_audio_orig)
            # os.remove(ref_audio_orig)

            non_silent_segs = silence.split_on_silence(aseg, min_silence_len=1000, silence_thresh=-50, keep_silence=500)
            non_silent_wave = AudioSegment.silent(duration=0)
            for non_silent_seg in non_silent_segs:
                non_silent_wave += non_silent_seg
            aseg = non_silent_wave

            audio_duration = len(aseg)
            if audio_duration > 15000:
                print("Audio is over 15s, clipping to only first 15s.")
                aseg = aseg[:15000]
            aseg.export(f.name, format="wav")
            ref_audio = f.name

        target_lang, _ = langid.classify(target_text)
        if prompt_text is None:
            print("No reference text provided, transcribing reference audio...")
            pipe = pipeline(
                "automatic-speech-recognition",
                model=openai_dir,
                torch_dtype=torch.float16,
                device=device,
            )
            prompt_text = pipe(
                ref_audio,
                chunk_length_s=30,
                batch_size=128,
                generate_kwargs={"task": "transcribe"},
                return_timestamps=False,
            )["text"].strip()
            print("Finished transcription")
            del pipe
            torch.cuda.empty_cache()
        prompt_lang, _ = langid.classify(prompt_text)
        print(f"prompt_language:{prompt_lang}")

        if len(prompt_text.encode('utf-8')) == len(prompt_text) and len(target_text.encode('utf-8')) == len(target_text):
            max_chars = 400-len(prompt_text.encode('utf-8'))
        else:
            max_chars = 300-len(prompt_text.encode('utf-8'))
        gen_text_batches = split_text_into_batches(target_text, max_chars=max_chars)
        print(f'ref_text\t{prompt_text}\n max_chars:{max_chars}')
        # gen_text_batches = text_list_normalize(gen_text_batches)
        audio_list = []
        for i, gen_text in enumerate(tqdm(gen_text_batches)):
            print(f'gen_text {i+1}', gen_text)
            recovered_audio = self.pipe.maskgct_inference(prompt_speech_path=ref_audio,
                                                                        prompt_text=prompt_text,
                                                                        target_text=gen_text,
                                                                        language=prompt_lang,
                                                                        target_language=target_lang)
            audio_list.append(recovered_audio)
        
        waveform = torch.from_numpy(np.concatenate(audio_list)).unsqueeze(0).unsqueeze(0)
        print(waveform.shape)
        res_audio = {
            "waveform":waveform,
            "sample_rate":24000
        }
        shutil.rmtree(tmp_dir)
        if not store_in_varm:
            self.pipe = None
            torch.cuda.empty_cache()
        return (res_audio,)

NODE_CLASS_MAPPINGS = {
    "MaskGCTNode": MaskGCTNode
}

SPLIT_WORDS = [
    "but", "however", "nevertheless", "yet", "still",
    "therefore", "thus", "hence", "consequently",
    "moreover", "furthermore", "additionally",
    "meanwhile", "alternatively", "otherwise",
    "namely", "specifically", "for example", "such as",
    "in fact", "indeed", "notably",
    "in contrast", "on the other hand", "conversely",
    "in conclusion", "to summarize", "finally"
]

import re

def split_text_into_batches(text, max_chars=200, split_words=SPLIT_WORDS):
    if len(text.encode('utf-8')) <= max_chars:
        return [text]
    if text[-1] not in ['。', '.', '!', '！', '?', '？']:
        text += '.'
        
    sentences = re.split('([。.!?！？])', text)
    sentences = [''.join(i) for i in zip(sentences[0::2], sentences[1::2])]
    
    batches = []
    current_batch = ""
    
    def split_by_words(text):
        words = text.split()
        current_word_part = ""
        word_batches = []
        for word in words:
            if len(current_word_part.encode('utf-8')) + len(word.encode('utf-8')) + 1 <= max_chars:
                current_word_part += word + ' '
            else:
                if current_word_part:
                    # Try to find a suitable split word
                    for split_word in split_words:
                        split_index = current_word_part.rfind(' ' + split_word + ' ')
                        if split_index != -1:
                            word_batches.append(current_word_part[:split_index].strip())
                            current_word_part = current_word_part[split_index:].strip() + ' '
                            break
                    else:
                        # If no suitable split word found, just append the current part
                        word_batches.append(current_word_part.strip())
                        current_word_part = ""
                current_word_part += word + ' '
        if current_word_part:
            word_batches.append(current_word_part.strip())
        return word_batches

    for sentence in sentences:
        if len(current_batch.encode('utf-8')) + len(sentence.encode('utf-8')) <= max_chars:
            current_batch += sentence
        else:
            # If adding this sentence would exceed the limit
            if current_batch:
                batches.append(current_batch)
                current_batch = ""
            
            # If the sentence itself is longer than max_chars, split it
            if len(sentence.encode('utf-8')) > max_chars:
                # First, try to split by colon
                colon_parts = sentence.split(':')
                if len(colon_parts) > 1:
                    for part in colon_parts:
                        if len(part.encode('utf-8')) <= max_chars:
                            batches.append(part)
                        else:
                            # If colon part is still too long, split by comma
                            comma_parts = re.split('[,，]', part)
                            if len(comma_parts) > 1:
                                current_comma_part = ""
                                for comma_part in comma_parts:
                                    if len(current_comma_part.encode('utf-8')) + len(comma_part.encode('utf-8')) <= max_chars:
                                        current_comma_part += comma_part + ','
                                    else:
                                        if current_comma_part:
                                            batches.append(current_comma_part.rstrip(','))
                                        current_comma_part = comma_part + ','
                                if current_comma_part:
                                    batches.append(current_comma_part.rstrip(','))
                            else:
                                # If no comma, split by words
                                batches.extend(split_by_words(part))
                else:
                    # If no colon, split by comma
                    comma_parts = re.split('[,，]', sentence)
                    if len(comma_parts) > 1:
                        current_comma_part = ""
                        for comma_part in comma_parts:
                            if len(current_comma_part.encode('utf-8')) + len(comma_part.encode('utf-8')) <= max_chars:
                                current_comma_part += comma_part + ','
                            else:
                                if current_comma_part:
                                    batches.append(current_comma_part.rstrip(','))
                                current_comma_part = comma_part + ','
                        if current_comma_part:
                            batches.append(current_comma_part.rstrip(','))
                    else:
                        # If no comma, split by words
                        batches.extend(split_by_words(sentence))
            else:
                current_batch = sentence
    
    if current_batch:
        batches.append(current_batch)
    
    return batches



