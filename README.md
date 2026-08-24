# sist2-script-clip
sist2 user script to augment images with [OpenAI CLIP](https://openai.com/research/clip) embeddings and generate tags with zero-shot classification.

Images, video thumbnails and images inside archives are all supported. When the original file cannot be
read (raw formats, or an index scanned somewhere else), the thumbnail sist2 stored is used instead.

Usage:

```
python run.py --help
```

| Option | Default | |
|---|---|---|
| `--clip-model` | `ViT-B/32` | CLIP variant. Only `ViT-B/32` has a published `.onnx` text encoder, which the web UI needs for embeddings search. |
| `--tags-file` | `general.txt` | One label per line, used for zero-shot classification. |
| `--num-tags` | `1` | Number of tags per document. `0` disables tagging. |
| `--color` | `#dcd7ff` | Color of the generated tags. |
| `--batch-size` | `32` | Images per inference batch. Raise it for more throughput, lower it if you run out of memory. |
| `--force` | | Discard the existing embeddings and process every document again. Implied when `--clip-model` changes. |

Only documents modified since the last run are processed. Progress is committed after every batch, so an
interrupted run resumes where it stopped.
