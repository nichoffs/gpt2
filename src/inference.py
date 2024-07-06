from gpt2 import GPT2
from config import GPT2Large
import argparse
import tiktoken

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='GPT2 Inference',
        description='Loads pre-trained weights and generates text. Enter the model size and a prompt.'
    )
    parser.add_argument('model_size', help='Size of the GPT-2 model (e.g., gpt2-large)')
    parser.add_argument('prompt', help='Input prompt for text generation')
    parser.add_argument('--num_return_sequences', type=int, default=2,
                        help='Number of return sequences (default: 2)')
    parser.add_argument('--max_length', type=int, default=100,
                        help='Maximum length of generated text (default: 100)')

    args = parser.parse_args()

    enc = tiktoken.get_encoding("gpt2")
    model = GPT2.build(args.model_size)

    generated_text = model.generate(
        args.prompt,
        enc,
        num_return_sequences=args.num_return_sequences,
        max_length=args.max_length
    )

    for text in generated_text:
        print(">", text)
