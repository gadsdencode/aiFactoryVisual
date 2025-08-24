import logging
from typing import Any, Dict, List
import random
import json

import torch
from datasets import Dataset, load_dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import PreTrainedTokenizer

from .config import DataConfig

logger = logging.getLogger(__name__)

def perturb_context(context: str, persona: str, intent: str) -> str:
    """Applies the Scenario-Perturbation Method to generate varied contexts."""
    perturbations = [
        lambda c: c.replace("busy professional", "retired hobbyist") if "Carla" in persona else c,
        lambda c: c.replace("30th birthday", "anniversary gift") if "Ben" in persona else c,
        lambda c: c.replace("dive watch", "chronograph watch") if "Alex" in persona else c,
        lambda c: f"{c} The user is now considering budget constraints." if "Carla" in persona else c,
        lambda c: f"{c} The user prefers a modern, tech-inspired design." if "Ben" in persona else c,
        lambda c: f"{c} The user is interested in limited-edition models." if "Alex" in persona else c,
    ]
    return random.choice(perturbations)(context)

def format_icdu_to_chat(example: Dict[str, Any], tokenizer: PreTrainedTokenizer) -> Dict[str, str]:
    """Converts an ICDU example to a chat-formatted string with optional perturbation."""
    required_fields = ["application_prompt", "ideal_response_final", "persona_archetype", "capability_layer", "context_summary", "user_intent"]
    if not all(field in example for field in required_fields):
        logger.warning(f"Skipping invalid ICDU: missing required fields. Found: {list(example.keys())}")
        return {"text": ""}

    # Extract fields
    prompt = example["application_prompt"]
    response = example["ideal_response_final"]
    persona = example["persona_archetype"]
    layer = example["capability_layer"]
    context = example["context_summary"]
    intent = example["user_intent"]

    # Apply perturbation with 50% probability to encourage application over recitation
    if random.random() < 0.5:
        context = perturb_context(context, persona, intent)
        # Adjust prompt slightly to reflect perturbed context
        if "budget constraints" in context:
            prompt = f"{prompt} I'm also looking to stay within a reasonable budget."
        elif "modern, tech-inspired design" in context:
            prompt = f"{prompt} I prefer something with a modern, techy look."
        elif "limited-edition models" in context:
            prompt = f"{prompt} I'm curious about limited-edition options."

    # Construct system prompt based on capability layer
    system_prompts = {
        "Foundational": "You are a factual and transparent assistant providing direct, unambiguous answers to user queries. Focus on clarity and practical information, ensuring responses are concise and helpful.",
        "Transformational": "You are an empathetic and insightful assistant. Guide users through complex decisions by reframing problems around their preferences and asking clarifying questions to ensure clarity in their decision-making process.",
        "Aspirational": "You are a knowledgeable and proactive partner. Affirm user choices, provide deep insights, and anticipate unstated needs to foster a long-term collaborative relationship."
    }
    system_content = system_prompts.get(layer, "You are a helpful assistant providing clear and accurate responses.")

    # Format as a chat message
    messages = [
        {"role": "system", "content": f"{system_content}\nContext: {context}\nUser Intent: {intent}\nPersona: {persona}"},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response}
    ]

    try:
        if getattr(tokenizer, "chat_template", None):
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
        else:
            # Fallback: manual formatting
            text = f"System: {messages[0]['content']}\nUser: {messages[1]['content']}\nAssistant: {messages[2]['content']}\n"
    except Exception as err:
        logger.warning(f"Chat template application failed ('{err}'). Using fallback formatting.")
        text = f"System: {messages[0]['content']}\nUser: {messages[1]['content']}\nAssistant: {messages[2]['content']}\n"

    return {"text": text}

def load_and_prepare_dataset(config: DataConfig, tokenizer: PreTrainedTokenizer) -> Dataset:
    """Loads, formats, and prepares the ICDU dataset for training."""
    logger.info(f"Loading dataset from {config.train_file} and {config.validation_file}")
    data_files = {
        "train": str(config.train_file),
        "validation": str(config.validation_file),
    }
    dataset = load_dataset("json", data_files=data_files)

    logger.info("Applying ICDU chat formatting to dataset...")
    formatted_dataset = dataset.map(
        lambda x: format_icdu_to_chat(x, tokenizer),
        remove_columns=list(dataset["train"].features)
    )

    # Filter out empty or invalid entries
    formatted_dataset = formatted_dataset.filter(lambda x: x["text"].strip() != "")

    # Tokenize to produce input_ids/attention_mask for Trainer
    logger.info("Tokenizing dataset...")
    tokenized = formatted_dataset.map(
        lambda batch: tokenizer(
            batch["text"],
            max_length=tokenizer.model_max_length,
            truncation=True,
            padding=False
        ),
        batched=True,
        remove_columns=["text"]
    )
    return tokenized
