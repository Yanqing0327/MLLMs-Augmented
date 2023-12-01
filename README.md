# MLLMs-Augmented Visual-Language Representation Learning
The official implementation of [MLLMs-Augmented Visual-Language Representation Learning](https://arxiv.org/pdf/2311.18765.pdf).

We will release the code soon!

## Abstract
Visual-language pre-training (VLP) has achieved remarkable success in multi-modal tasks, largely attributed to the availability of large-scale image-text datasets. In this work, we demonstrate that multi-modal large language models (MLLMs) can enhance visual-language representation learning by improving data quality. Our approach is simple, utilizing MLLMs to extend multiple captions for each image. To prevent the bias introduced by MLLMs' hallucinations and intrinsic caption styles, we propose "text shearing" to maintain the same length for extended captions as that of the original captions. In image-text retrieval, our method consistently obtains 5.6 ~ 35.0\% and 16.8 ~ 46.1\% improvement on R@1 under the fine-tuning and zero-shot settings, respectively. Notably, we obtain zero-shot results that are comparable to fine-tuning on target datasets, which encourages more exploration of the versatile use of MLLMs.
