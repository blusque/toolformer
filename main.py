from transformers.pipelines.text_generation import TextGenerationPipeline

def get_pretrained_generator() -> TextGenerationPipeline:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from transformers.pipelines import pipeline

    tokenizer = AutoTokenizer.from_pretrained(r"dmayhem93/toolformer_v0_epoch2")
    model = AutoModelForCausalLM.from_pretrained(
        r"dmayhem93/toolformer_v0_epoch2",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).cuda()
    generator = pipeline(
        "text-generation", model=model, tokenizer=tokenizer, device=0
    )
    return generator    

def main():
    generator = get_pretrained_generator()
    response = generator("Hello from toolformer!", max_new_tokens=10)
    print(response)


if __name__ == "__main__":
    main()
