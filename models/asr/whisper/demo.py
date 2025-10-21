import torch

from transformers import WhisperProcessor, WhisperForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPastAndCrossAttentions    

import os
import argparse
from datasets import load_dataset

import tcim_lite as tcim

HOUMO_TARGET = os.getenv("HOUMO_TARGET")

class HoumoWhisperEncoder(torch.nn.Module):
    def __init__(self, hg_encoder, encoder_path):
        super().__init__()
        self.conv1 = hg_encoder.conv1
        self.conv2 = hg_encoder.conv2
        weight_manager = tcim.runtime.WeightManager(0)
        option1 = tcim.runtime.Option(weight_manager)
        self.encoder = tcim.runtime.load(encoder_path, option=option1)

    def forward(
        self,
        input_features,
        attention_mask=None,
        head_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):   
        input_features = input_features.numpy()
        self.encoder.set_input(self.encoder.get_input_name(0), input_features)
        self.encoder.run()
        self.encoder.sync()
        hidden_states = torch.tensor(self.encoder.get_output(self.encoder.get_output_name(0)).numpy())
        hidden_states = torch.randn([1, 1500, 1024], dtype=torch.float16)
        encoder_states = None
        all_attentions = None
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
        )

class HoumoWhisperDecoder(torch.nn.Module):
    def __init__(self, hg_decoder, decoder_path):
        super().__init__()
        weight_manager = tcim.runtime.WeightManager(0)
        option1 = tcim.runtime.Option(weight_manager)
        self.decoder = tcim.runtime.load(decoder_path, option=option1)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        inputs_embeds=None,
        position_ids=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        cache_position=None,
    ):
        input_ids = input_ids.numpy()
        encoder_hidden_states = encoder_hidden_states.numpy()
        self.decoder.set_input(self.decoder.get_input_name(0), input_ids)
        self.decoder.set_input(self.decoder.get_input_name(1), encoder_hidden_states)
        self.decoder.run()
        self.decoder.sync()
        hidden_states = torch.tensor(self.decoder.get_output(self.decoder.get_output_name(0)).numpy())
        next_cache = None
        all_hidden_states = None
        all_self_attns = None
        all_cross_attentions = None
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
        )

def get_args() -> argparse.Namespace:
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_dir',
        dest='model_dir',
        type=str,
        default="whisper-medium",
        help='raw model dir',
    )
    parser.add_argument(
        '--encoder_path',
        dest='encoder_path',
        type=str,
        default=os.path.join('output', HOUMO_TARGET, "encoder.hmm"),
        help='houmo encoder model path',
    )
    parser.add_argument(
        '--decoder_path',
        dest='decoder_path',
        type=str,
        default=os.path.join('output', HOUMO_TARGET, "decoder.hmm"),
        help='houmo decoder model path',
    )
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    # load model and processor
    processor = WhisperProcessor.from_pretrained(args.model_dir)
    model = WhisperForConditionalGeneration.from_pretrained(args.model_dir)
    model.config.forced_decoder_ids = None
    model.model.encoder = HoumoWhisperEncoder(model.model.encoder, args.encoder_path)
    model.model.decoder = HoumoWhisperDecoder(model.model.decoder, args.decoder_path)

    # load dummy dataset and read audio files
    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    sample = ds[0]["audio"]
    input_features = processor(sample["array"], sampling_rate=sample["sampling_rate"], return_tensors="pt").input_features 

    # generate token ids
    predicted_ids = model.generate(input_features)
    # decode token ids to text
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=False)
    print(transcription)
    # ['<|startoftranscript|><|en|><|transcribe|><|notimestamps|> Mr. Quilter is the apostle of the middle classes and we are glad to welcome his gospel.<|endoftext|>']

    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
    print(transcription)
    # [' Mr. Quilter is the apostle of the middle classes and we are glad to welcome his gospel.']