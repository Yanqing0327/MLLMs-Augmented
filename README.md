We provide code for generating captions using MiniGPT-4, Qwen-VL, Otter, and LLaVA-1.5.
You need to download the model's weight following the official implementation.
We provide the corresponding repository here:
MiniGPT-4: https://github.com/Vision-CAIR/MiniGPT-4.git
Qwen-VL: https://github.com/QwenLM/Qwen-VL.git
Otter: https://github.com/Luodian/Otter.git
LLaVA-1.5: https://github.com/haotian-liu/LLaVA.git

You can generate captions by running:

```
conda env create -f environment.yml
conda activate xxxxx
sh ./model_name/generate.sh
```

Before generated captions, you may need to split the json file. And there is code for processing the json file in the folder './json_process'

After these operations, you can conduct the visual-langugage pretraining in a standard pipeline.
