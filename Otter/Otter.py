import mimetypes
import os
from io import BytesIO
from typing import Union
import cv2
import requests
import torch
import transformers
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor
from tqdm import tqdm
import sys
import datetime
import json
import argparse
import numpy as np
sys.path.append("../../src")

from otter_ai import OtterForConditionalGeneration


# Disable warnings
requests.packages.urllib3.disable_warnings()

# ------------------- Utility Functions -------------------


def get_content_type(file_path):
    content_type, _ = mimetypes.guess_type(file_path)
    return content_type


# ------------------- Image Handling Functions -------------------


def get_image(url: str) -> Union[Image.Image, list]:
    if not url.strip():  # Blank input, return a blank Image
        return Image.new("RGB", (224, 224))  # Assuming 224x224 is the default size for the model. Adjust if needed.
    elif "://" not in url:  # Local file
        content_type = get_content_type(url)
    else:  # Remote URL
        content_type = requests.head(url, stream=True, verify=False).headers.get("Content-Type")

    if "image" in content_type:
        if "://" not in url:  # Local file
            return Image.open(url)
        else:  # Remote URL
            return Image.open(requests.get(url, stream=True, verify=False).raw)
    else:
        raise ValueError("Invalid content type. Expected image.")


# ------------------- OTTER Prompt and Response Functions -------------------


def get_formatted_prompt(prompt: str) -> str:
    return f"<image>User: {prompt} GPT:<answer>"


def get_response(image, prompt: str, model=None, image_processor=None) -> str:
    input_data = image

    if isinstance(input_data, Image.Image):
        if input_data.size == (224, 224) and not any(input_data.getdata()):  # Check if image is blank 224x224 image
            vision_x = torch.zeros(1, 1, 1, 3, 224, 224, dtype=next(model.parameters()).dtype)
        else:
            vision_x = image_processor.preprocess([input_data], return_tensors="pt")["pixel_values"].unsqueeze(1).unsqueeze(0)
    else:
        raise ValueError("Invalid input data. Expected PIL Image.")

    lang_x = model.text_tokenizer(
        [
            get_formatted_prompt(prompt),
        ],
        return_tensors="pt",
    )

    model_dtype = next(model.parameters()).dtype

    vision_x = vision_x.to(dtype=model_dtype)
    lang_x_input_ids = lang_x["input_ids"]
    lang_x_attention_mask = lang_x["attention_mask"]

    generated_text = model.generate(
        vision_x=vision_x.to(model.device),
        lang_x=lang_x_input_ids.to(model.device),
        attention_mask=lang_x_attention_mask.to(model.device),
        max_new_tokens=512,
        num_beams=3,
        no_repeat_ngram_size=3,
    )
    parsed_output = (
        model.text_tokenizer.decode(generated_text[0])
        .split("<answer>")[-1]
        .lstrip()
        .rstrip()
        .split("<|endofchunk|>")[0]
        .lstrip()
        .rstrip()
        .lstrip('"')
        .rstrip('"')
    )
    return parsed_output


# ------------------- Main Function -------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--piece-number", type=int, default=0, help="")
    parser.add_argument("--force-continue", action='store_true', help="")
    args = parser.parse_args()
    load_bit = "bf16"
    precision = {}
    if load_bit == "bf16":
        precision["torch_dtype"] = torch.bfloat16
    elif load_bit == "fp16":
        precision["torch_dtype"] = torch.float16
    elif load_bit == "fp32":
        precision["torch_dtype"] = torch.float32
    model = OtterForConditionalGeneration.from_pretrained("luodian/OTTER-Image-MPT7B", device_map="sequential", **precision)
    model.text_tokenizer.padding_side = "left"
    tokenizer = model.text_tokenizer
    image_processor = transformers.CLIPImageProcessor()
    model.eval()

    input_json = f'/json_path/partition_{args.piece_number}.json'
    print(f'load from {input_json}')
    out_json = input_json[:-5]+f"_otter"+".json"
    image_list = []
    image_pack_list = []
    number=0
    with open(input_json, 'r') as json_file:
        # 加载JSON数据并解析为Python对象
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
    continue_point = 0
    if os.path.exists(out_json):
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
    if args.force_continue and continue_point!=0:
        continue_point = continue_point+1
    if os.path.exists(out_json):
        with open(out_json, 'a') as outfile:
            for i in range(continue_point,len(image_pack_list)):
                if i == 0:
                    outfile.write('[')
                image_list = image_pack_list[i]
                question_list = ['Describe in English:']*len(image_list)
                imgs = [get_image(img) for img in image_list]
                try:
                    vision_x = image_processor.preprocess(imgs, return_tensors="pt")["pixel_values"].unsqueeze(1).unsqueeze(1)
                except Exception as e:
                    print(f"Error processing image at list {i}: {str(e)}")
                    continue

                prompts = [f"<image> User: {question} GPT: <answer>" for question in question_list]
                lang_x = model.text_tokenizer(prompts, return_tensors="pt", padding=True)
                generated_text = model.generate(
                    vision_x=vision_x.to(model.device, dtype=torch.float16),
                    lang_x=lang_x["input_ids"].to('cuda'),
                    attention_mask=lang_x["attention_mask"].to('cuda', dtype=torch.float16),
                    max_new_tokens=30,
                    num_beams=1,
                    no_repeat_ngram_size=3,
                )
                total_output = []
                for j in range(len(generated_text)):
                    output = model.text_tokenizer.decode(generated_text[j])
                    output = [x for x in output.split(' ') if not x.startswith('<')]
                    out_label = output.index('GPT:')
                    output = ' '.join(output[out_label + 1:])
                    output = output.split("<|endofchunk|>")[0]
                    output = output.split(".")[0]
                    total_output.append(output)
                
                for k in range(len(image_list)):
                    data = {'image': image_list[k], 'caption': total_output[k]}
                    json.dump(data, outfile)
                
                    if i==(len(image_pack_list)-1) and k==(len(image_list)-1):
                        outfile.write(']')
                    else:
                        outfile.write(',')
                print(i,len(image_pack_list))
                current_time = datetime.datetime.now()
                print(current_time)
    else:
        with open(out_json, 'w') as outfile:
            for i in range(len(image_pack_list)):
                if i == 0:
                    outfile.write('[')
                image_list = image_pack_list[i]
                question_list = ['Describe in English:']*len(image_list)
                imgs = [get_image(img) for img in image_list]
                vision_x = image_processor.preprocess(imgs, return_tensors="pt")["pixel_values"].unsqueeze(1).unsqueeze(1)
                prompts = [f"<image> User: {question} GPT: <answer>" for question in question_list]
                lang_x = model.text_tokenizer(prompts, return_tensors="pt", padding=True)
                generated_text = model.generate(
                    vision_x=vision_x.to(model.device, dtype=torch.float16),
                    lang_x=lang_x["input_ids"].to('cuda'),
                    attention_mask=lang_x["attention_mask"].to('cuda', dtype=torch.float16),
                    max_new_tokens=30,
                    num_beams=1,
                    no_repeat_ngram_size=3,
                )
                total_output = []
                for j in range(len(generated_text)):
                    output = model.text_tokenizer.decode(generated_text[j])
                    output = [x for x in output.split(' ') if not x.startswith('<')]
                    out_label = output.index('GPT:')
                    output = ' '.join(output[out_label + 1:])
                    output = output.split("<|endofchunk|>")[0]
                    output = output.split(".")[0]
                    total_output.append(output)
                
                for k in range(len(image_list)):
                    data = {'image': image_list[k], 'caption': total_output[k]}
                    json.dump(data, outfile)
                
                    if i==(len(image_pack_list)-1) and k==(len(image_list)-1):
                        outfile.write(']')
                    else:
                        outfile.write(',')
                print(i,len(image_pack_list))
                current_time = datetime.datetime.now()
                print(current_time)