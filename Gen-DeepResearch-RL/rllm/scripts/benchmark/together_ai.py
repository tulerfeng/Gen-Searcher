import argparse
import concurrent.futures
import json
import os
import time

import pandas as pd
from together import Together
from tqdm import tqdm

from rllm.data.utils import TestDataset, TrainDataset, fetch_live_code_bench_system_prompt, load_dataset
from rllm.rewards.code_reward import RewardCodeFn, extract_code_from_model
from rllm.rewards.reward_types import RewardConfig, RewardInput, RewardType

HUMANEVALPLUS_PROMPT = "Think step by step: please provide an efficient and self-contained Python script that solves the following problem in a markdown code block:\n\n"


def generate_response(client, prompt, model="deepseek-ai/DeepSeek-R1"):
    # append the prompt to the messages
    messages = [{"role": "user", "content": prompt}]

    max_retries = 3
    attempts = 0

    while attempts < max_retries:
        try:
            stream = client.chat.completions.create(model=model, messages=messages, temperature=0.6, top_p=0.95, max_tokens=32768, stream=True)
            full_response = ""
            for chunk in stream:
                full_response += chunk.choices[0].delta.content or ""
            return full_response.strip()
        except Exception as e:
            attempts += 1
            print(f"Error encountered (attempt {attempts}/{max_retries}): {e}")
            if attempts >= max_retries:
                return ""
            time.sleep(1)


def preload_data(dataset_name):
    upper_ds_name = dataset_name.upper()
    print(f"Loading dataset {upper_ds_name}...")
    if upper_ds_name in TestDataset.Code.__members__:
        ds = TestDataset.Code[upper_ds_name]
    elif upper_ds_name in TrainDataset.Code.__members__:
        ds = TrainDataset.Code[upper_ds_name]
    else:
        # throw error if dataset is not found
        raise ValueError(f"Dataset {dataset_name} not found.")

    dataset = load_dataset(ds)
    return dataset


def generation_loop(client, dataset_name, model, output_dir, n=1, skip_rewards=False):
    skip_generation = False
    if not os.path.exists(os.path.join(output_dir, "responses.parquet")):
        dataset = preload_data(dataset_name)
        df = pd.json_normalize(dataset)
    else:
        print(f"Loading existing responses from {os.path.join(output_dir, 'responses.parquet')}")
        df = pd.read_parquet(os.path.join(output_dir, "responses.parquet"))
        dataset = df.to_dict(orient="records")
        skip_generation = True
    all_responses = []
    all_scores = []

    reward = RewardCodeFn(RewardConfig)

    def process_item(args):
        idx, item = args
        prompt = item["problem"]
        if dataset_name != "humanevalplus":
            prompt = fetch_live_code_bench_system_prompt(prompt)
        response_lst = []
        scores_lst = []
        for i in range(n):
            if skip_generation:
                response = item["responses"][i]
            else:
                if dataset_name == "humanevalplus":
                    prompt = HUMANEVALPLUS_PROMPT + prompt
                response = generate_response(client, prompt, model=model)

            if "def solve():" in response:
                extracted_code = extract_code_from_model(response)
                # check if extracted_code ends with solve()
                if extracted_code and not extracted_code.endswith("solve()"):
                    extracted_code += "\nsolve()\n"
                response = f"```python\n{extracted_code}```"

            response_lst.append(response)
            score = None
            if not skip_rewards:
                if dataset_name == "humanevalplus":
                    tests = item["tests"]
                else:
                    tests = item["tests"].tolist() if not isinstance(item["tests"], list) else item["tests"]
                input_obj = RewardInput(problem="", problem_type=RewardType.CODE, model_response=response, metadata=tests, data_source=dataset_name)
                score = reward(input_obj).reward
            scores_lst.append(score)
        return idx, response_lst, scores_lst

    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
        results = list(tqdm(executor.map(process_item, enumerate(dataset)), total=len(dataset)))
        for idx, response_lst, scores_lst in results:
            all_responses.append((idx, response_lst))
            all_scores.append((idx, scores_lst))

    # order the lists by idx
    all_responses = [x[1] for x in sorted(all_responses, key=lambda x: x[0])]
    all_scores = [x[1] for x in sorted(all_scores, key=lambda x: x[0])]

    # output the overall accuracy
    # Calculate and display pass@1 and pass@n accuracy
    if not skip_rewards:
        pass_at_1 = sum([1 for scores in all_scores if any(score > 0 for score in scores[:1])]) / len(all_scores)
        pass_at_n = sum([1 for scores in all_scores if any(score > 0 for score in scores)]) / len(all_scores)
        print(f"Pass@1: {pass_at_1:.4f}")
        print(f"Pass@{n}: {pass_at_n:.4f}")

    df["responses"] = all_responses
    df["scores"] = all_scores
    os.makedirs(output_dir, exist_ok=True)
    df.to_parquet(os.path.join(output_dir, "responses.parquet"))

    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(all_scores, f)


def main():
    parser = argparse.ArgumentParser(description="Generate a response from the OpenAI reasoning model given a prompt.")
    # arg for specifying dataset name
    parser.add_argument("--dataset-name", type=str, required=True, help="Name of the dataset to use.")
    # arg for specifying output dir
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory to save the results.")
    args = parser.parse_args()

    client = Together()

    try:
        generation_loop(client, args.dataset_name, "deepseek-ai/DeepSeek-R1", args.output_dir, n=8)
    except Exception as e:
        print(f"An error occurred: {e}")
        # print stack trace
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
