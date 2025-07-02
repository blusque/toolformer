import torch
from transformers.pipelines.text_generation import TextGenerationPipeline
from accelerate import PartialState
from vllm import LLM, SamplingParams

def set_debug_params():
    """
    Set debug environment variables for CUDA and vLLM.
    """
    import os
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["VLLM_DEBUG"] = "1"
    os.environ["VLLM_LOG_LEVEL"] = "DEBUG"
    os.environ["VLLM_LOG_FILE"] = "vllm_debug.log"
    os.environ["VLLM_MAX_BATCH_SIZE"] = "1"
    os.environ["VLLM_MAX_MODEL_LEN"] = "2048"

def get_pretrained_generator():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from transformers.pipelines import pipeline

    # tokenizer = AutoTokenizer.from_pretrained(r"dmayhem93/toolformer_v0_epoch2")
    # model = AutoModelForCausalLM.from_pretrained(
    #     r"dmayhem93/toolformer_v0_epoch2",
    #     torch_dtype=torch.float16,
    #     low_cpu_mem_usage=True,
    # ).cuda()
    llm = LLM(
        model="facebook/opt-125m",
        tensor_parallel_size=1,
        max_model_len=2048,
        device="cuda",
    )
    sampling_params = SamplingParams(
        temperature=0.1,
        top_p=0.95,
        top_k=50,
        max_tokens=512,
        stop=["\n"],
        do_sample=True,
        repetition_penalty=1.0,
    )
    prompts = [
        "Hello, how are you?",
        "What is the capital of France?",
        "Tell me a joke.",
        "What is the meaning of life?",
        "Explain quantum computing in simple terms.",
        "What is the weather like today?",
    ]
    outputs = llm.generate(prompts, sampling_params)
    return outputs
    # for prompt, output in zip(prompts, outputs):
    #     print(f"Prompt: {prompt}\nOutput: {output.outputs[0].text}\n")
    # tokenizer = AutoTokenizer.from_pretrained(r"deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    # model = AutoModelForCausalLM.from_pretrained(r"deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", torch_dtype=torch.float16, low_cpu_mem_usage=True).cuda()
    # param_amount = sum(p.numel() for p in model.parameters())
    # print(f"Total parameters: {param_amount / 1e6:.2f}M")
    # distributed_state = PartialState()
    # generator = pipeline(
    #     "text-generation", model=model, tokenizer=tokenizer, device=distributed_state.device,
    # )
    # return generator    

def main():
    set_debug_params()
    outputs = get_pretrained_generator()
    for output in outputs:
        print(f"Prompt: {output.prompt}\nOutput: {output.outputs[0].text}\n")
    # generator = get_pretrained_generator()
    # response = generator("Hello from toolformer!", max_new_tokens=100)
    # print(response)


if __name__ == "__main__":
    main()
