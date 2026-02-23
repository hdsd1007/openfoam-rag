import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
# from langchain_community.llms import HuggingFacePipeline
from langchain_google_genai import ChatGoogleGenerativeAI




def load_generator_llm(max_tokens: int = 2048):
    """
    Load Gemini Pro for RAG evaluation.
    Uses same API key as generator.

    Args:
        max_tokens: Maximum output tokens. Default 2048 for single-query,
                    use 8192 for report generation.
    """

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY not found.\n"
            "In Kaggle: Add it as a secret in Notebook Settings > Add-ons > Secrets.\n"
            "Locally: export GOOGLE_API_KEY='your-key'"
        )

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",  # More rigorous for evaluation
        temperature=0.2,  # Deterministic for judging
        max_output_tokens=max_tokens,
        google_api_key=api_key
    )

    return llm

# def load_generator_llm():

#     model_id = "Qwen/Qwen2.5-3B-Instruct"

#     tokenizer = AutoTokenizer.from_pretrained(model_id)

#     model = AutoModelForCausalLM.from_pretrained(
#         model_id,
#         device_map="auto",
#         torch_dtype=torch.float16
#     )

#     pipe = pipeline(
#         "text-generation",
#         model=model,
#         tokenizer=tokenizer,
#         max_new_tokens=1024,
#         temperature=0.0,          # small creativity
#         return_full_text=False,
#         do_sample = False,
#         pad_token_id=tokenizer.eos_token_id,
#         repetition_penalty=1.1
#     )

#     return HuggingFacePipeline(pipeline=pipe)
