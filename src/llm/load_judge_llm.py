import os
from langchain_google_genai import ChatGoogleGenerativeAI


def load_judge_llm():
    """
    Load Gemini Pro for RAG evaluation.
    Uses same API key as generator.
    """
    
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY not found.\n"
            "In Kaggle: Add it as a secret in Notebook Settings > Add-ons > Secrets.\n"
            "Locally: export GOOGLE_API_KEY='your-key'"
        )
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",  # Stronger model than generator to avoid self-preference bias
        temperature=0.0,  # Deterministic for judging
        max_output_tokens=4096,  # Needs headroom for thinking tokens (Flash) + JSON output
        google_api_key=api_key
    )
    
    return llm

# import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
# from langchain_community.llms import HuggingFacePipeline


# def load_judge_llm():

#     model_id = "HuggingFaceTB/SmolLM3-3B"

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
#         max_new_tokens=600,
#         temperature=0.0,        # deterministic
#         return_full_text=False
#     )

#     return HuggingFacePipeline(pipeline=pipe)
