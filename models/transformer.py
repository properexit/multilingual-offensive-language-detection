from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import LoraConfig, get_peft_model


def load_transformer(model_name, num_labels, peft_type=None):
    """
    Loads tokenizer and transformer model.
    Optionally applies LoRA if peft_type == "lora".
    """

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels
    )

    # if LoRA is requested, wrap the model
    if peft_type == "lora":
        print("Using LoRA adaptation")

        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["query", "value"],  # attention projection layers
            lora_dropout=0.1,
            bias="none",
            task_type="SEQ_CLS"
        )

        model = get_peft_model(model, lora_config)

        # show how many parameters are actually trainable
        model.print_trainable_parameters()

    return tokenizer, model