import os
import argparse
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from transformers.modeling_outputs import BaseModelOutputWithPast

import tcim_lite as tcim

HOUMO_TARGET = os.getenv("HOUMO_TARGET")

def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery: {query}'

class HoumoQwen2Model(torch.nn.Module):
    def __init__(self, gte_path):
        super().__init__()
        weight_manager = tcim.runtime.WeightManager(0)
        option1 = tcim.runtime.Option(weight_manager)
        self.gte = tcim.runtime.load(gte_path, option=option1)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.LongTensor] = None,
        is_causal: Optional[bool] = True,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        input_ids = input_ids
        attention_mask = attention_mask
        attention_mask_flip = attention_mask.flip(1)
        pad_size = 2048 - input_ids.size(1)
        input_ids = F.pad(input_ids, (0, pad_size), mode='constant', value=0).numpy()
        attention_mask = F.pad(attention_mask, (0, pad_size), mode='constant', value=0).numpy()
        attention_mask_flip = F.pad(attention_mask_flip, (0, pad_size), mode='constant', value=0).numpy()
        self.gte.set_input(self.gte.get_input_name(0), input_ids)
        self.gte.set_input(self.gte.get_input_name(1), attention_mask)
        self.gte.set_input(self.gte.get_input_name(2), attention_mask_flip)
        self.gte.run()
        self.gte.sync()
        hidden_states = self.gte.get_output(self.gte.get_output_name(0)).numpy()
        next_cache = None
        all_hidden_states = None
        all_self_attns = None
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

def get_args() -> argparse.Namespace:
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--tokenizer_dir',
        dest='tokenizer_dir',
        type=str,
        default="gte-Qwen2-1.5B-instruct",
        help='tokenizer dir',
    )
    parser.add_argument(
        '--model_path',
        dest='model_path',
        type=str,
        default=os.path.join('output', HOUMO_TARGET, "gte.hmm"),
        help='houmo gte model path',
    )
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    max_length = 2048
    model = HoumoQwen2Model(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir, trust_remote_code=True)

    # Each query must come with a one-sentence instruction that describes the task
    task = 'Given a web search query, retrieve relevant passages that answer the query'
    queries = [
        get_detailed_instruct(task, 'how much protein should a female eat'),
        get_detailed_instruct(task, 'summit define')
    ]
    # No need to add instruction for retrieval documents
    documents = [
        "As a general guideline, the CDC's average requirement of protein for women ages 19 to 70 is 46 grams per day. But, as you can see from this chart, you'll need to increase that if you're expecting or training for a marathon. Check out the chart below to see how much protein you should be eating each day.",
        "Definition of summit for English Language Learners. : 1  the highest point of a mountain : the top of a mountain. : 2  the highest level. : 3  a meeting or series of meetings between the leaders of two or more governments."
    ]
    input_texts = queries + documents
    # Tokenize the input texts
    batch_dict = tokenizer(input_texts, max_length=max_length, padding=True, truncation=True, return_tensors='pt')
    # print(model)

    outputs = model(**batch_dict)

    embeddings = torch.tensor(outputs.last_hidden_state)

    # normalize embeddings
    embeddings = F.normalize(embeddings, p=2, dim=1)
    scores = (embeddings[:2] @ embeddings[2:].T) * 100
    # [[70.00666809082031, 8.184867858886719], [14.62420654296875, 77.71405792236328]]
