
# Nuclues 1B Alpha1

<p align="center">
    <img src="https://github.com/prp-e/nucleus/raw/main/nucleus-logo.png" width=256 height=256>
</p>

## What is Nucleus?

Nucleus is a small language model based on Mistral (actually, the trimmed untrained version you can find [here](https://huggingface.co/lmlab/lmlab-mistral-1b-untrained)) and trained in different steps. First, we've pretrained it on TinyStories dataset, then [TinyTextBooks](https://huggingface.co/datasets/nampdn-ai/tiny-textbooks) to make it a more specific model. This model is just a _proof of concept_ at this point, but showed good promises in early tests. So with proper training, can be a good product over time!

## Inference

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/prp-e/nucleus/blob/main/nucleus_1b_inference.ipynb)

First you need to install `transformers` and `accelerate` libraries in order to run this model. Then, you basically have to run the following code:

```python

from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import torch

model_name_or_id = "NucleusOrg/Nucleus-1B-alpha-1"

model = AutoModelForCausalLM.from_pretrained(model_name_or_id, torch_dtype=torch.float16, device_map="cuda")
tokenizer = AutoTokenizer.from_pretrained(model_name_or_id)

prompt = "### Lesson: Python Programming 101\n### Introduction\n"

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

generation_config = GenerationConfig(
    do_sample=True,
    top_k=1,
    temperature=0.9,
    max_new_tokens=500,
    repetition_penalty=1.5,
    pad_token_id=tokenizer.eos_token_id
)

outputs = model.generate(**inputs, generation_config=generation_config)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

```

__Prompt Format__: This model does not have a specific prompt format, but the best results could be achieved with a _textbook_ type of format like:

```
### Chapter 1: Elon Musk and Iron Man
Elon met Tony at a Cafe in Monaco, then they had a conversation about
```

You also can try something like this: 

```
Question: Who are you?
Answer:
```

But since the model isn't made for chat/question answering, the result won't be good enough. 

__Repetition Penalty__: Since most of these models like to repeat themselves, just keep that number there. You can increase or decrease it based on your liking,but keep in mind that a number lower than 1 makes the model _super repetitive_. 

## Known Issues

* Since we only had 420k rows of data, a lot of information are missing on this model. Since mentioned earlier in this very model card, it's a _proof of concept_ model.
* You probably may test it with coding. Let's say that the model is terrible at coding. We may release a coding optimized model as soon as possible. 

## Our Team

* Muhammadreza Haghiri ([[X (formerly Twitter)](https://twitter.com/haghiri_ai) - Website](https://haghiri75.com/en) - [Github](https://github.com/prp-e) - [LinkedIn](https://www.linkedin.com/in/muhammadreza-haghiri-1761325b))
* Mahi Mohrechi ([Website](https://mohrechi-portfolio.vercel.app/) - [Github](https://github.com/f-mohrechi) - [LinkedIn](https://www.linkedin.com/in/faeze-mohrechi/))

## Special Thanks

* LMLabs for providing 1B untrained model. 
* Mistral Team for providing the best open source base model ever.
* _Sina Rashidi_, who translated Alpaca dataset to Persian.
* [Jupyto](https://jupyto.com) team for providing our infrastructure.