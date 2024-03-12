import argparse
import itertools
import json
import os
import random
import time
from functools import partial

import torch
from pycocoevalcap.eval import COCOEvalCap
from pycocotools.coco import COCO
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
import datetime
import re

class CaptionDataset(torch.utils.data.Dataset):

    def __init__(self, test, prompt):
        # self.images = json.load(open(test))['images']
        # print(train)
        # print(test)
        with open(test, 'r') as json_file:
            data = json.load(json_file)
        self.images = data
        # self.images = json.load(open(test))

        self.prompt = prompt

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]['image']

        return {
            'path': image_path,
            'input_text': self.prompt.format(image_path)
        }


def collate_fn(inputs, tokenizer):
    path = [_['path'] for _ in inputs]
    input_texts = [_['input_text'] for _ in inputs]
    input_tokens = tokenizer(input_texts,
                             return_tensors='pt',
                             padding='longest')

    return path, input_tokens.input_ids, input_tokens.attention_mask


class InferenceSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, size):
        self._size = int(size)
        assert size > 0
        self._rank = torch.distributed.get_rank()
        self._world_size = torch.distributed.get_world_size()
        self._local_indices = self._get_local_indices(size, self._world_size,
                                                      self._rank)

    @staticmethod
    def _get_local_indices(total_size, world_size, rank):
        shard_size = total_size // world_size
        left = total_size % world_size
        shard_sizes = [shard_size + int(r < left) for r in range(world_size)]

        begin = sum(shard_sizes[:rank])
        end = min(sum(shard_sizes[:rank + 1]), total_size)
        return range(begin, end)

    def __iter__(self):
        yield from self._local_indices

    def __len__(self):
        return len(self._local_indices)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='piece')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--few-shot', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument("--piece-number", type=int, default=0, help="")
    args = parser.parse_args()

    torch.distributed.init_process_group(
        backend='nccl',
        world_size=int(os.getenv('WORLD_SIZE', '1')),
        rank=int(os.getenv('RANK', '0')),
    )

    torch.cuda.set_device(int(os.getenv('LOCAL_RANK', 0)))

    prompt = '<img>{}</img>Express emotional or artistic response:'

    model = AutoModelForCausalLM.from_pretrained(
        '/model_path', device_map='cuda', trust_remote_code=True).eval()

    tokenizer = AutoTokenizer.from_pretrained('/model_path',
                                              trust_remote_code=True)
    tokenizer.padding_side = 'left'
    tokenizer.pad_token_id = tokenizer.eod_id
    print(args.batch_size)
    random.seed(args.seed)
    input_json = '/json_path/'+f'partition_{args.piece_number}.json'
    out_json = input_json[:-5]+f"_qianwen"+".json"
    dataset = CaptionDataset(
        test=input_json,
        prompt=prompt
    )
    coco_karpathy_test_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        sampler=InferenceSampler(len(dataset)),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=partial(collate_fn, tokenizer=tokenizer),
    )

    if os.path.exists(out_json):
        with open(out_json, 'r') as file:
            json_text = file.read()
        substring = "/image_path/"
        last_position = json_text.rfind(substring)
        substring_after_last = json_text[last_position:]
        first_jpg_position = substring_after_last.find("jpg")
        last_image_path = json_text[last_position:last_position+first_jpg_position+3]
        continue_point = 0
        for k, (path, input_ids,
                    attention_mask) in tqdm(enumerate(coco_karpathy_test_loader)):
            flag=0
            image_list = path
            for j in range(len(image_list)):
                if image_list[j] == last_image_path:
                    continue_point = k
                    flag=1
                    break
            if flag==1:
                break
        index = json_text.find(image_list[0])
        json_text = json_text[:index-11]
        with open(out_json, 'w') as file:
            file.write(json_text)

    if os.path.exists(out_json):
        with open(out_json, 'a') as outfile:
            for k, (path, input_ids,
                    attention_mask) in tqdm(enumerate(coco_karpathy_test_loader)):
                if k<continue_point:
                    continue
                if k == 0:
                    outfile.write('[')
                pred = model.generate(
                    input_ids=input_ids.cuda(),
                    attention_mask=attention_mask.cuda(),
                    do_sample=False,
                    num_beams=1,
                    max_new_tokens=30,
                    min_new_tokens=8,
                    length_penalty=0,
                    num_return_sequences=1,
                    use_cache=True,
                    pad_token_id=tokenizer.eod_id,
                    eos_token_id=tokenizer.eod_id,
                )
                captions = []
                for _ in pred:
                    captions.append(tokenizer.decode(_[input_ids.size(1):].cpu(),
                                    skip_special_tokens=True).strip().split('.')[0])
                
                current_time = datetime.datetime.now()
                print(k,current_time)
            
                for i in range(len(captions)):
                    data = {'image': path[i], 'caption': captions[i]}
                    json.dump(data, outfile)
                    if len(captions)<args.batch_size and i==(len(captions)-1):
                        outfile.write(']')
                    else:
                        outfile.write(',')
            
            torch.distributed.barrier()
    else:
        with open(out_json, 'w') as outfile:
            for k, (path, input_ids,
                    attention_mask) in tqdm(enumerate(coco_karpathy_test_loader)):
                if k == 0:
                    outfile.write('[')
                pred = model.generate(
                    input_ids=input_ids.cuda(),
                    attention_mask=attention_mask.cuda(),
                    do_sample=False,
                    num_beams=1,
                    max_new_tokens=30,
                    min_new_tokens=8,
                    length_penalty=0,
                    num_return_sequences=1,
                    use_cache=True,
                    pad_token_id=tokenizer.eod_id,
                    eos_token_id=tokenizer.eod_id,
                )
                captions = []
                for _ in pred:
                    captions.append(tokenizer.decode(_[input_ids.size(1):].cpu(),
                                    skip_special_tokens=True).strip().split('.')[0])
                
                current_time = datetime.datetime.now()
                print(k,current_time)
            
                for i in range(len(captions)):
                    data = {'image': path[i], 'caption': captions[i]}
                    json.dump(data, outfile)
                    if len(captions)<args.batch_size and i==(len(captions)-1):
                        outfile.write(']')
                    else:
                        outfile.write(',')
            
            torch.distributed.barrier()