import time

from simple_evals.gpqa_eval import GPQAEval

from utils.samplers.ollama_sampler import OllamaSampler
from utils.utils import load_config


def run_gpqa_eval():
    start_time = time.time()
    # Load the configuration file
    config = load_config("src/config/config.yaml")

    # Initialize Ollama sampler (wrapper around Ollama chat)
    ollama_sampler = OllamaSampler(model_name=config["MODEL_NAME"])

    # Instantiate the GPQAEval class for evaluation
    gpqa_eval = GPQAEval(n_repeats=1, variant="diamond", num_examples=1)

    # Run GPQA evaluation
    results = gpqa_eval(ollama_sampler)

    end_time = time.time()
    elapsed_seconds = end_time - start_time
    minutes, seconds = divmod(
        elapsed_seconds, 60
    )  # Convert execution time to minutes and seconds

    # The returned results is an EvalResult which includes a list of SingleEvalResult
    # and aggregated metrics. Print metrics:
    print("Overall Evaluation Metrics:")
    print(results.metrics)
    print(f"Total Execution Time: {int(minutes)} min {seconds:.2f} sec")


if __name__ == "__main__":
    run_gpqa_eval()
