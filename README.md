Install with Python 3.11 or later:

```bash
python -m pip install -e .
tc-norm
```

Choose `LLM config → Initialize llm.json` to enter your model and API settings, then `Run → Run all agents`. Select a PDF directory or search query and enter a run name.

Outputs are saved in `results/<run_name>/`. Example Mapping output:

```json
{
  "title": "Example paper",
  "rule_version": "prompt",
  "items": [
    {
      "original_label": "CD8 T cells",
      "evidence": ["The population expresses CD8."],
      "mapped_label": "CD8+ T",
      "mapped_slots": {
        "lineage": {"module": "CD8+", "semantic_value": "CD8-positive T cells"},
        "function": {"module": null, "semantic_value": null},
        "migration": {"module": null, "semantic_value": null},
        "migration_subscript": {"module": null, "semantic_value": null},
        "differentiation": {"module": null, "semantic_value": null},
        "differentiation_subscript": {"module": null, "semantic_value": null},
        "antigen": {"module": null, "semantic_value": null}
      }
    }
  ]
}
```
