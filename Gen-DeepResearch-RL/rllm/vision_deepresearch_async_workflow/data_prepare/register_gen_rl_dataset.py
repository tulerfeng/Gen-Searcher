#!/usr/bin/env python3
"""
Load data from a JSON file and register it to the rllm dataset registry.
Extract prompt and gt_image for RL training.
"""

from rllm.data.dataset import DatasetRegistry
import json
import random
import argparse
import os


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Load gen_data_demo.json and register dataset to rllm"
    )
    
    parser.add_argument(
        "--json_path", 
        type=str, 
        required=True,
        help="Path to the gen_data_demo.json file"
    )
    parser.add_argument(
        "--register_name", 
        type=str, 
        default="Vision-DeepResearch-Gen",
        help="Name for the registered dataset (default: Vision-DeepResearch-Gen)"
    )
    parser.add_argument(
        "--train_ratio", 
        type=float, 
        default=0.9,
        help="Ratio for train split (default: 0.9)"
    )
    parser.add_argument(
        "--random_seed", 
        type=int, 
        default=42,
        help="Random seed for shuffling (default: 42)"
    )
    
    return parser.parse_args()


def load_json(file_path):
    """Load a JSON file."""
    print(f"Loading JSON file: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Total samples loaded: {len(data)}")
    return data


def extract_data(json_data):
    """
    Extract fields required for training from input JSON.
    Fields: id, prompt, gt_image
    """
    extracted_data = []
    skipped_count = 0
    
    for item in json_data:
        try:
            # Extract required fields
            data_id = item.get('id', None)
            prompt = item.get('prompt', None)
            
            # Open-source convention: gt_image is stored at the top level
            gt_image = item.get('gt_image', None)
            
            # Validate required fields
            if data_id is None or not prompt or not gt_image:
                skipped_count += 1
                continue
            
            # Validate that the gt_image file exists
            if not os.path.exists(gt_image):
                print(f"Warning: GT image not found: {gt_image}, skipping id={data_id}")
                skipped_count += 1
                continue
            
            extracted_data.append({
                'id': data_id,
                'prompt': prompt,
                'gt_image': gt_image
            })
            
        except Exception as e:
            print(f"Error processing item: {e}")
            skipped_count += 1
            continue
    
    print(f"Successfully extracted: {len(extracted_data)} samples")
    print(f"Skipped: {skipped_count} samples")
    
    return extracted_data


def process_data(data_list, start_id=0):
    """
    Convert extracted data into the registration format.
    Format: {id, question, gt_image}
    """
    output = []
    for i, item in enumerate(data_list):
        x = {
            'id': str(start_id + i),
            'question': item['prompt'],
            'gt_image': item['gt_image']
        }
        output.append(x)
    return output


def register_dataset(data, register_name, split):
    """Register a dataset to DatasetRegistry."""
    registry_dataset = DatasetRegistry.register_dataset(register_name, data, split)
    print(f"Registered {len(data)} samples to DatasetRegistry as '{register_name}' ({split})")
    return registry_dataset


def main():
    # 1. Parse arguments
    args = parse_args()
    
    print(f"{'='*60}")
    print(f"JSON Path: {args.json_path}")
    print(f"Register Name: {args.register_name}")
    print(f"Train Ratio: {args.train_ratio}")
    print(f"Random Seed: {args.random_seed}")
    print(f"{'='*60}\n")
    
    # 2. Check file existence
    if not os.path.exists(args.json_path):
        print(f"Error: JSON file not found: {args.json_path}")
        return 1
    
    # 3. Load JSON data
    json_data = load_json(args.json_path)
    
    # 4. Extract required fields
    extracted_data = extract_data(json_data)
    
    if len(extracted_data) == 0:
        print("Error: No valid data extracted!")
        return 1
    
    # 5. Shuffle data
    random.seed(args.random_seed)
    random.shuffle(extracted_data)
    
    # 6. Split train/test by ratio
    split_idx = int(len(extracted_data) * args.train_ratio)
    train_data = extracted_data[:split_idx]
    test_data = extracted_data[split_idx:]
    
    print(f"\nTrain samples: {len(train_data)}")
    print(f"Test samples: {len(test_data)}")
    
    # 7. Transform data
    train_output = process_data(train_data, start_id=0)
    test_output = process_data(test_data, start_id=0)
    
    # 8. Print an example
    print(f"\n{'='*60}")
    print("Sample train data (first item):")
    print(json.dumps(train_output[0], indent=2, ensure_ascii=False))
    print(f"{'='*60}\n")
    
    # 9. Register dataset
    register_dataset(train_output, args.register_name, "train")
    register_dataset(test_output, args.register_name, "test")
    
    print(f"\n{'='*60}")
    print("Dataset registration completed successfully!")
    print(f"{'='*60}")
    
    return 0


if __name__ == "__main__":
    exit(main())
