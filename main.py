

from transformers import AutoTokenizer
import transformers
import torch

model = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model, use_auth_token=True)

#
from transformers import pipeline
llama_pipeline = pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto"
)
#
def get_llama_response(prompt: str) -> None:
    sequences = llama_pipeline(
        prompt,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eox_token_id,
        max_length=256,
    )
    print("ChatBot: ", sequences[0]["generated_text"])

prompt = "I like breaking bad, what about you?"
get_llama_response(prompt)

