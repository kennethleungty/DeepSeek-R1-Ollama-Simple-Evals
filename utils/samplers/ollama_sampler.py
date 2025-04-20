import ollama


class OllamaSampler:
    def __init__(self, model_name=None, temperature=0):
        self.model_name = model_name
        self.temperature = temperature

    def __call__(self, prompt_messages):
        prompt_text = prompt_messages[-1]["content"]
        response = ollama.chat(
            model=self.model_name, 
            messages=[{"role": "user", "content": prompt_text}],
            options={"temperature": self.temperature}
        )
        response_content = response["message"]["content"]

        return response_content

    def _pack_message(self, content, role):
        return {"role": role, "content": content}
