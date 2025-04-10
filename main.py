import time

from dotenv import load_dotenv

from simple_evals.gpqa_eval import GPQAEval
from simple_evals.math_eval import MathEval
from simple_evals.sampler.chat_completion_sampler import ChatCompletionSampler
from utils.samplers.ollama_sampler import OllamaSampler
from utils.utils import load_config

# Load env variables
load_dotenv()


def run_eval():
    start_time = time.time()
    config = load_config("config/config.yaml")

    # Initialize Ollama sampler (wrapper around Ollama chat)
    ollama_sampler = OllamaSampler(model_name=config["MODEL_NAME"])

    # Choose which evaluation class to use based on EVAL_BENCHMARK
    eval_benchmark = config["EVAL_BENCHMARK"]
    print(f">>> Running {eval_benchmark} evaluation")
    if eval_benchmark == "gpqa":
        eval_class = GPQAEval
        eval_kwargs = {
            "n_repeats": config["EVAL_N_REPEATS"],
            "num_examples": config["EVAL_N_EXAMPLES"],
            "variant": config["GPQA_VARIANT"],
        }
    elif eval_benchmark == "math":
        # Define equality checker (LLM as judge)
        equality_checker = ChatCompletionSampler(model="gpt-4-turbo-preview")

        eval_class = MathEval
        eval_kwargs = {
            "equality_checker": equality_checker,
            "n_repeats": config["EVAL_N_REPEATS"],
            "num_examples": config["EVAL_N_EXAMPLES"],
            "split": config["MATH_VARIANT"],
        }
    else:
        raise ValueError(
            f"Unknown EVAL_BENCHMARK '{eval_benchmark}'. Must be 'gpqa' or 'math'."
        )

    # Instantiate and run the appropriate eval
    evaluator = eval_class(**eval_kwargs)
    results = evaluator(ollama_sampler)

    end_time = time.time()
    elapsed_seconds = end_time - start_time
    minutes, seconds = divmod(elapsed_seconds, 60)

    # The returned results is an EvalResult which includes a list of SingleEvalResult
    # and aggregated metrics
    print(">>>> Overall Evaluation Metrics:")
    print(results.metrics)
    print(">>>> Score:")
    print(results.score)
    print(f">>>> Total Execution Time: {int(minutes)} min {seconds:.2f} sec")


if __name__ == "__main__":
    run_eval()
