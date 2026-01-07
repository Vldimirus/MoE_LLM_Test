# General Assistant

**Expert ID:** `general`
**Version:** 0.1.0-test
**Type:** Test Model (Random Weights)

## Architecture

- **Model Type:** Transformer-based Expert
- **Parameters:** 2,092,032 (7.98 MB)
- **Layers:** 2
- **Hidden Size:** 256
- **Attention Heads:** 4
- **Vocabulary Size:** 1000
- **Max Sequence Length:** 512

## Status

⚠️ **This is a TEST model with random weights.**

It has NOT been trained and will generate random/nonsensical output.
This model is for testing infrastructure and integration only.

## Usage

```python
from models.expert import ExpertModel

# Load model
checkpoint = torch.load("models/experts/general/model.pt")
model = ExpertModel(**checkpoint['config'])
model.load_state_dict(checkpoint['model_state_dict'])

# Generate (will be random!)
output = model.generate(input_ids, max_length=50)
```

## Training

To train this model, use:

```bash
python scripts/train_expert.py --expert-id general --data path/to/data.jsonl
```
