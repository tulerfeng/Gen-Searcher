# Data Format (Parquet Only)

## Supported Fields

- `question`: Required. A non-empty string (must include `image_id`).
- `answer`: Required. A string.
- `images`: Optional. A list; each element is an **absolute local file path** string.
- Any other columns will be ignored. If any required column is missing, an error will be raised.

## Minimal Example (Logical Structure; use the same column names when writing to Parquet)

```text
question: "image_id:1 Question: What animals are shown?"
answer: "duck and lion"
images: ["/path/to/image"]
```