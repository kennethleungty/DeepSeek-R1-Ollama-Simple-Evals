# ==================================
# This script is intended for testing purposes, designed to mock the behavior of a sampler used in GPQA evaluation process.
# It provides a simple way to simulate responses from a model without needing to call the actual model.
# The MockSampler class is initialized with a fixed response and will always return that response when called.
# ==================================
from dataclasses import dataclass, field
from typing import Any, Dict, List

from simple_evals.gpqa_eval import GPQAEval

Message = dict[str, Any]  # keys role, content
MessageList = list[Message]

class MockSampler:
    def __init__(self, fixed_response: str):
        """
        Initializes the mock sampler to always return a specific string.
        Args:
            fixed_response: The exact string the sampler should return.
        """
        self.fixed_response = fixed_response
        print(f"[MockSampler Setup] Will always return: '{self.fixed_response}'")

    def _pack_message(self, content: str, role: str) -> Dict[str, str]:
        """Mimics the message packing used by GPQAEval."""
        return {"content": content, "role": role}

    def __call__(self, prompt_messages: List[Dict[str, str]]) -> str:
        """Returns the predefined fixed response, ignoring the prompt."""
        print(f"[MockSampler Called] Returning fixed response: '{self.fixed_response}'")
        return self.fixed_response

# --- Test Execution Example ---
# This block shows how to use GPQAEval with the MockSampler.
# Run this part within your project where all dependencies are met.
if __name__ == "__main__":

    NUM_MULTI_EXAMPLES = 15 # Number of examples to run each test on
    VARIANT = "diamond"    # GPQA variant to use

    print(f"--- Initializing Evaluator (variant='{VARIANT}', {NUM_MULTI_EXAMPLES} examples) ---")
    evaluator_multi = GPQAEval(variant=VARIANT, num_examples=NUM_MULTI_EXAMPLES, n_repeats=1)

    if not evaluator_multi.examples:
         print(f"Error: Evaluator has no examples loaded (requested {NUM_MULTI_EXAMPLES}). Check CSV path or available data.")
         # Exit or handle error appropriately if needed
    else:
        actual_examples_loaded = len(evaluator_multi.examples)
        print(f"--- Evaluator initialized with {actual_examples_loaded} examples ---")

        # --- Create Mock Samplers ---
        mock_sampler_a = MockSampler(fixed_response="ANSWER:A")
        mock_sampler_b = MockSampler(fixed_response="ANSWER:B")
        mock_sampler_c = MockSampler(fixed_response="ANSWER:C")
        mock_sampler_d = MockSampler(fixed_response="ANSWER:D")
        mock_sampler_malformed = MockSampler(fixed_response="Maybe B? Not sure.")

        # --- Run Tests on the Same Multi-Example Set ---

        print(f"\n--- Running Test A: Mock Response 'Answer: A' on {actual_examples_loaded} examples ---")
        result_a = evaluator_multi(mock_sampler_a)
        # The score will be the fraction of times 'A' was the correct answer in the sample
        print(f"Result A -> Score: {result_a.score:.2f}, Examples: {len(result_a.htmls)}, Metrics: {result_a.metrics}")

        print(f"\n--- Running Test B: Mock Response 'Answer: B' on {actual_examples_loaded} examples ---")
        result_b = evaluator_multi(mock_sampler_b)
        # The score will be the fraction of times 'B' was the correct answer in the sample
        print(f"Result B -> Score: {result_b.score:.2f}, Examples: {len(result_b.htmls)}, Metrics: {result_b.metrics}")

        print(f"\n--- Running Test C: Mock Response 'Answer: C' on {actual_examples_loaded} examples ---")
        result_c = evaluator_multi(mock_sampler_c)
        # The score will be the fraction of times 'C' was the correct answer in the sample
        print(f"Result C -> Score: {result_c.score:.2f}, Examples: {len(result_c.htmls)}, Metrics: {result_c.metrics}")

        print(f"\n--- Running Test D: Mock Response 'Answer: D' on {actual_examples_loaded} examples ---")
        result_d = evaluator_multi(mock_sampler_d)
        # The score will be the fraction of times 'D' was the correct answer in the sample
        print(f"Result D -> Score: {result_d.score:.2f}, Examples: {len(result_d.htmls)}, Metrics: {result_d.metrics}")

        print(f"\n--- Running Test Malformed: Mock Response Malformed on {actual_examples_loaded} examples ---")
        result_malformed = evaluator_multi(mock_sampler_malformed)
        # The score should ideally be 0.0 as the answer cannot be extracted
        print(f"Result Malformed -> Score: {result_malformed.score:.2f}, Examples: {len(result_malformed.htmls)}")
