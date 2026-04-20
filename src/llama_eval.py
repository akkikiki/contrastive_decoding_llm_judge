from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

import json
import argparse
import tqdm

from contrastive_decoding import ContrastiveLlamaForCausalLM, ContrastiveQwen2ForCausalLM

if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--prompt_fp', type=str, default='prompts/summeval/con_detailed_range.txt')
    argparser.add_argument('--save_fp', type=str, default='results/llama_con_detailed.json')
    argparser.add_argument('--summeval_fp', type=str, default='data/summeval.json')
    argparser.add_argument('--model', type=str, default='meta-llama/Llama-3.1-8B-Instruct')
    argparser.add_argument('--assistant_model', type=str, default='meta-llama/Llama-3.2-3B-Instruct')
    argparser.add_argument('--batch_size', type=int, default=16)
    argparser.add_argument('--decoding', choices=['contrastive', 'greedy', 'sampling'],
                          default='sampling', help='Decoding strategy to use')
    argparser.add_argument('--contrastive_lamb', type=float, default=1.0)
    argparser.add_argument('--contrastive_asst_temperature', type=float, default=1.0)
    args = argparser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    NUM_RETURN_SEQUENCES = 20 if args.decoding == 'sampling' else 1

    if args.decoding == 'contrastive':
        assistant_model = AutoModelForCausalLM.from_pretrained(args.assistant_model, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
        assistant_tokenizer = AutoTokenizer.from_pretrained(args.assistant_model)

        if "Qwen" in args.model:
            model = ContrastiveQwen2ForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
            # Qwen 3B and 7B have vocab size mismatch; align to tokenizer length
            model.config.vocab_size = len(tokenizer)
            model.config.get_text_config().vocab_size = len(tokenizer)
            model.resize_token_embeddings(len(tokenizer))
            model.config.contrastive_lamb = args.contrastive_lamb
            model.config.contrastive_asst_temperature = args.contrastive_asst_temperature

            assistant_model.config.vocab_size = len(assistant_tokenizer)
            assistant_model.config.get_text_config().vocab_size = len(assistant_tokenizer)
            assistant_model.resize_token_embeddings(len(assistant_tokenizer))
        else:
            model = ContrastiveLlamaForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
            model.config.contrastive_lamb = args.contrastive_lamb
            model.config.contrastive_asst_temperature = args.contrastive_asst_temperature
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")

    if torch.cuda.is_available():
        print('using cuda')
        model = model.to('cuda')
        if args.decoding == 'contrastive':
            assistant_model = assistant_model.to('cuda')

    summeval = json.load(open(args.summeval_fp))
    print(args.prompt_fp)
    prompt = open(args.prompt_fp).read()

    ct, ignore = 0, 0
    BATCH_SIZE = args.batch_size
    new_json = []
    print(len(summeval))

    for i in tqdm.tqdm(range(0, len(summeval), BATCH_SIZE)):
        batch = summeval[i:i + BATCH_SIZE]
        batch_messages = []

        for instance in batch:
            source = instance['source']
            system_output = instance['system_output']
            cur_prompt = prompt.replace('{{Document}}', source).replace('{{Summary}}', system_output)
            instance['prompt'] = cur_prompt
            batch_messages.append([{"role": "system", "content": cur_prompt}])

        batch_prompts = [
            tokenizer.apply_chat_template(messages, tokenize=False)
            for messages in batch_messages
        ]

        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=2048
        ).to(model.device)
        print(inputs.keys())

        if args.decoding == 'contrastive':
            outputs = model.generate_contrastive(
                **inputs,
                max_new_tokens=64,
                do_sample=False,
                assistant_model=assistant_model,
                return_dict_in_generate=True,
                output_scores=True,
            )
        elif args.decoding == 'greedy':
            outputs = model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=False,
                num_return_sequences=1,
                return_dict_in_generate=True,
                output_scores=True,
            )
        else:  # sampling
            outputs = model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=True,
                temperature=2.0,
                top_p=1.0,
                num_return_sequences=NUM_RETURN_SEQUENCES,
                return_dict_in_generate=True,
                output_scores=True,
            )

        sequences = outputs.sequences
        logits = outputs.scores

        if args.decoding == 'contrastive':
            print(f"raw returned sequences shape: {sequences.shape}")
            sequences = sequences.reshape(len(batch), 1, -1)
        else:
            sequences = sequences.reshape(len(batch), NUM_RETURN_SEQUENCES, -1)
        print(sequences.shape)

        for idx, instance in enumerate(batch):
            instance_sequences = sequences[idx]
            all_responses = []
            all_logits = []

            score_token_ids = [tokenizer.convert_tokens_to_ids(str(i)) for i in range(10)]

            for seq_idx, sequence in enumerate(instance_sequences):
                decoded_text = tokenizer.decode(
                    sequence[inputs.input_ids[idx].shape[0]:],
                    skip_special_tokens=True
                ).replace('assistant', '').strip()

                all_responses.append(decoded_text)

                if logits and len(logits) > 0:
                    if args.decoding == 'sampling':
                        # For sampling, logits shape is [batch_size * num_return_sequences, vocab_size]
                        first_token_logits = logits[0][idx * NUM_RETURN_SEQUENCES + seq_idx]
                    else:
                        # For greedy/contrastive, logits shape is [batch_size, vocab_size]
                        first_token_logits = logits[0][idx]

                    score_logits = [first_token_logits[token_id].item() for token_id in score_token_ids]
                    all_logits.append(score_logits)
                else:
                    all_logits.append([0.0] * 10)

            instance['all_responses'] = all_responses
            instance['all_logits'] = all_logits
            new_json.append(instance)
            ct += 1

    print('ignored total', ignore)
    with open(args.save_fp, 'w') as f:
        json.dump(new_json, f, indent=4)
