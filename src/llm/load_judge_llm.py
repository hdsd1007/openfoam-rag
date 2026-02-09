import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_community.llms import HuggingFacePipeline


def load_judge_llm():

    model_id = "HuggingFaceTB/SmolLM3-3B"

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.float16
    )

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=600,
        temperature=0.0,        # deterministic
        return_full_text=False
    )

    return HuggingFacePipeline(pipeline=pipe)
