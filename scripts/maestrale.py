import argparse

import torch
from transformers import TextStreamer, AutoModelForCausalLM, AutoTokenizer, GenerationConfig


class Maestrale:
    def __init__(self, device="cuda:0"):
        # Pointless to run on CPU, check if GPU is available otherwise refuse
        if not torch.cuda.is_available():
            raise Exception("No GPU detected. Please make sure your hardware matches LLM requirements before running this script.")
        self.tokenizer = AutoTokenizer.from_pretrained("mii-llm/maestrale-chat-v0.3-alpha")
        self.model = AutoModelForCausalLM.from_pretrained("mii-llm/maestrale-chat-v0.3-alpha",
                                                          load_in_8bit=True,
                                                          device_map=device,
                                                          cache_dir="./models/cache")
        self.gen_config = GenerationConfig(
            do_sample=True,
            temperature=0.7,
            repetition_penalty=1.2,
            top_k=50,
            top_p=0.95,
            max_new_tokens=200,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.convert_tokens_to_ids("<|im_end|>"),
            skip_prompt=True,
            skip_special_tokens=True,
            output_scores=False,
            return_dict_in_generate=False
        )

        self.chat_template = [
            {"role": "system",
             "content": "Sei un assistente utile. Rispondi alle domande in modo conciso, senza spiegazioni. Se l'informazione non é specificata, rispondi solamente 'non specificato'."},
            {"role": "user", "content": f""}
        ]

    def generate_answer(self, question: str, context: str, stream=False, confidence=False):
        template = self.chat_template
        template[1]["content"] = f"Data la seguente nota clinica: '{context}'\nRispondi alla seguente domanda: {question}."
        return self._generate_(chat_template=template, stream=stream, confidence=confidence)

    def __get_confidence__(self, scores, ids):
        # OPTION 1: calculate softmax, then access element [index]
        # normalized_logits_per_step = [torch.nn.functional.softmax(scores[i], dim=1)[0][index] for i, index in enumerate(ids[0])]
        # OPTION 2: directly calculate exp of element [index] and divide by sum of exps --> 1 division instead of 30K
        normalized_logits_per_step = [torch.exp(scores[i][0][index]) / torch.sum(torch.exp(scores[i][0])) for i, index in enumerate(ids[0])]
        return torch.Tensor(normalized_logits_per_step).mean()

    def _generate_(self, chat_template, stream=False, confidence=False):
        with torch.no_grad(), torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
            template = self.tokenizer.apply_chat_template(chat_template, tokenize=False, add_generation_prompt=True)
            inputs = self.tokenizer(template, return_tensors="pt").to("cuda")
            config = self.gen_config
            if stream:
                streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
                _ = self.model.generate(
                    **inputs,
                    streamer=streamer,
                    generation_config=config
                )
                return None
            else:
                if confidence:
                    config.output_scores = True,
                    config.return_dict_in_generate = True
                    output = self.model.generate(**inputs, generation_config=config)
                    confidence_score = self.__get_confidence__(scores=output["scores"],
                                                               ids=output["sequences"][:, inputs["input_ids"].shape[1]:])
                    return {"text": self.tokenizer.batch_decode(output["sequences"][:, inputs["input_ids"].shape[1]:],
                                                                skip_special_tokens=True)[0],
                            "confidence": confidence_score.item()
                            }
                else:
                    output = self.model.generate(**inputs, generation_config=config)
                    return {"text": self.tokenizer.batch_decode(output[:, inputs["input_ids"].shape[1]:],
                                                                skip_special_tokens=True)[0],
                            "confidence": None
                            }
