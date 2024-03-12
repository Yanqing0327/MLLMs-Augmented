import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import datetime
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image


def eval_model(args):
    disable_torch_init()
    model_path = args.model_path
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)
    print('load model')
    json_path = '/json_path/'
    input_json = json_path+f'partition_{args.piece_number}.json'
    print(input_json)
    out_json = input_json[:-5]+f"_llava"+".json"

    image_list = []
    image_pack_list = []
    number=0
    with open(input_json, 'r') as json_file:
        data = json.load(json_file)
    for i in range(len(data)):
        image_list.append(data[i]['image'])
        number+=1
        if number==48:
            image_pack_list.append(image_list)
            image_list = []
            number=0
        if i == (len(data)-1) and len(image_list)!=0:
            image_pack_list.append(image_list)

    if os.path.exists(out_json):
        continue_point=0
        with open(out_json, 'r') as file:
            json_text = file.read()
        substring = "/image_path/"
        last_position = json_text.rfind(substring)
        substring_after_last = json_text[last_position:]
        first_jpg_position = substring_after_last.find("jpg")
        last_image_path = json_text[last_position:last_position+first_jpg_position+3]
        for i in range(len(image_pack_list)):
            flag=0
            image_list = image_pack_list[i]
            for j in range(len(image_list)):
                if image_list[j] == last_image_path:
                    continue_point = i
                    flag=1
                    break
            if flag==1:
                break
        index = json_text.find(image_pack_list[continue_point][0])
        json_text = json_text[:index-11]
        with open(out_json, 'w') as file:
            file.write(json_text)
    continue_point = continue_point+1
    prompt = 'Describe <image> in English:'
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
    if os.path.exists(out_json):
        with open(out_json, 'a') as outfile:
            for i in range(continue_point,len(image_pack_list)):
                if i == 0:
                    outfile.write('[')
                image_url_list = image_pack_list[i]
                image_list = []
                for img_path in image_url_list:
                    try:
                        image = image_processor.preprocess(Image.open(img_path), return_tensors='pt')['pixel_values'][0].unsqueeze(0).half().cuda()
                        image_list.append(image)
                    except Exception as e:
                        print(f"Error processing image at path {img_path}: {str(e)}")
                        continue
                new_input_ids = input_ids.repeat(len(image_list), 1)
                with torch.inference_mode():
                    output_ids = model.generate(
                        new_input_ids,
                        images=image_list,
                        do_sample=True if args.temperature > 0 else False,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        num_beams=args.num_beams,
                        max_new_tokens=30,
                        use_cache=True)
                caption = []
                for _ in output_ids:
                    caption.append(tokenizer.decode(_[input_ids.size(1):].cpu(),
                                            skip_special_tokens=True).strip().split('.')[0])
                for j in range(len(image_list)):
                    image_url = image_url_list[j]
                    data = {'image': image_url, 'caption': caption[j]}
                    json.dump(data, outfile)
                    
                    if i==(len(image_pack_list)-1) and j==(len(image_url_list)-1):
                        outfile.write(']')
                    else:
                        outfile.write(',')

                print(f'{i},{len(image_pack_list)}')
                current_time = datetime.datetime.now()
                print(current_time)
    else:
        with open(out_json, 'w') as outfile:
            for i in range(len(image_pack_list)):
                if i == 0:
                    outfile.write('[')
                image_url_list = image_pack_list[i]
                image_list = []
                for img_path in image_url_list:
                    try:
                        image = image_processor.preprocess(Image.open(img_path), return_tensors='pt')['pixel_values'][0].unsqueeze(0).half().cuda()
                        image_list.append(image)
                    except Exception as e:
                        print(f"Error processing image at path {img_path}: {str(e)}")
                        continue
                new_input_ids = input_ids.repeat(len(image_list), 1)
                with torch.inference_mode():
                    output_ids = model.generate(
                        new_input_ids,
                        images=image_list,
                        do_sample=True if args.temperature > 0 else False,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        num_beams=args.num_beams,
                        max_new_tokens=30,
                        use_cache=True)
                caption = []
                for _ in output_ids:
                    caption.append(tokenizer.decode(_[input_ids.size(1):].cpu(),
                                            skip_special_tokens=True).strip().split('.')[0])
                for j in range(len(image_list)):
                    image_url = image_url_list[j]
                    data = {'image': image_url, 'caption': caption[j]}
                    json.dump(data, outfile)
                    
                    if i==(len(image_pack_list)-1) and j==(len(image_url_list)-1):
                        outfile.write(']')
                    else:
                        outfile.write(',')

                print(f'{i},{len(image_pack_list)}')
                current_time = datetime.datetime.now()
                print(current_time)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/model_path")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--piece-number", type=int, default=0, help="")
    args = parser.parse_args()

    eval_model(args)
