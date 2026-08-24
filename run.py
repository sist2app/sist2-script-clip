from io import BytesIO
from sys import stderr

import typer
import clip
import torch
from PIL import Image

from sist2 import Sist2Index, print_progress

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_ID = 1

# Elasticsearch only maps these embedding sizes (schema/mappings.json in sist2)
SUPPORTED_SIZES = (384, 512, 768, 1024)

# The only CLIP variant with a published .onnx text encoder for the web UI
ONNX_MODEL = "ViT-B/32"
ONNX_URL = "https://raw.githubusercontent.com/sist2app/sist2-models/main/clip/models/clip-vit-base-patch32-q8.onnx"

# Zero-shot classification is considerably more accurate with a prompt than with a bare label
TAG_TEMPLATE = "a photo of a {}"

print(f"Using compute device {DEVICE}")


def load_tag_embeddings(tag_file, model):
    with open(tag_file) as f:
        tags = list(dict.fromkeys(line.strip() for line in f if line.strip()))

    text_tokenized = clip.tokenize([TAG_TEMPLATE.format(tag) for tag in tags]).to(DEVICE)
    with torch.no_grad():
        tag_embeddings = model.encode_text(text_tokenized)
    tag_embeddings /= tag_embeddings.norm(dim=-1, keepdim=True)

    print(f"Pre-computed embeddings for {len(tags)} tags")

    return tag_embeddings, tags


def get_thumbnail(index, doc_id):
    # Sist2Index.get_thumbnail() returns an arbitrary one of the thumbnails a video has
    index.cur.execute("SELECT data FROM thumbnail WHERE id=? AND num=0", (doc_id,))
    row = index.cur.fetchone()

    return row[0] if row else None


def load_image(index, doc, preprocess):
    if not doc.parent and doc.mime and doc.mime.startswith("image/"):
        try:
            with Image.open(doc.path) as image:
                return preprocess(image)
        except Exception as e:
            # Formats Pillow cannot read (raw, heif, psd) and originals that are no longer
            # where they were scanned from still have a thumbnail in the index
            print(f"Could not load image {doc.rel_path} ({e}), falling back to the thumbnail", file=stderr)

    tn = get_thumbnail(index, doc.id)
    if not tn:
        raise Exception("Could not find thumbnail")

    with Image.open(BytesIO(tn)) as image:
        return preprocess(image)


def tag_document(doc, similarity, tags, num_tags, color):
    top_n = similarity.topk(min(num_tags, len(tags))).indices.tolist()
    top_n_tags = [f"clip.{tags[i]}.{color}" for i in top_n]

    if "tag" not in doc.json_data:
        doc.json_data["tag"] = top_n_tags
    else:
        doc.json_data["tag"] = [
            *(t for t in doc.json_data["tag"] if not t.startswith("clip.")),
            *top_n_tags
        ]


def process_batch(index, batch, model, tag_embeddings, tags, num_tags, color):
    docs, images = zip(*batch)

    with torch.no_grad():
        embeddings = model.encode_image(torch.stack(images).to(DEVICE))

    if num_tags > 0:
        similarity = (embeddings / embeddings.norm(dim=-1, keepdim=True)) @ tag_embeddings.T

        for doc, doc_similarity in zip(docs, similarity):
            tag_document(doc, doc_similarity, tags, num_tags, color)
            index.update_document(doc)

    encoded = embeddings.cpu().numpy().astype("<f4")

    for doc, embedding in zip(docs, encoded):
        index.upsert_embedding(doc.id, 0, None, MODEL_ID, embedding.tobytes())
        print(f"Generated embeddings for {doc.rel_path}")

    index.set("clip_last_id", docs[-1].id)
    index.commit()


def register_model(index, clip_model, size):
    if size not in SUPPORTED_SIZES:
        raise typer.BadParameter(f"{clip_model} has {size}-dimension embeddings, but sist2 only maps "
                                 f"{', '.join(map(str, SUPPORTED_SIZES))}")

    url = ONNX_URL if clip_model == ONNX_MODEL else None
    if url is None:
        print(f"No .onnx text encoder is published for {clip_model}, so embeddings search will not be "
              f"available in the web UI. Use {ONNX_MODEL} for that.", file=stderr)

    index.register_model(
        id=MODEL_ID,
        name="CLIP",
        url=url,
        path=f"idx_{size}.clip",
        size=size,
        type="flat"
    )


def main(index_file, clip_model: str = "ViT-B/32", tags_file: str = "general.txt", num_tags: int = 1,
         color: str = "#dcd7ff", batch_size: int = 32, force: bool = False):
    model, preprocess = clip.load(clip_model, device=DEVICE)

    tag_embeddings, tags = load_tag_embeddings(tags_file, model)

    index = Sist2Index(index_file)

    register_model(index, clip_model, tag_embeddings.shape[-1])

    # Embeddings of a different model are not comparable with the ones this run generates
    previous_model = index.get("clip_model")
    if previous_model is not None and previous_model != clip_model:
        force = True

    if force:
        index.cur.execute("DELETE FROM embedding WHERE model_id=?", (MODEL_ID,))
        index.set("clip_version", 0)
        index.set("clip_last_id", 0)

    # Written before any work, not after it: a run that died partway through would otherwise be
    # taken for a model change by the next one, and throw away everything it had committed
    index.set("clip_model", clip_model)

    # Only consider documents that were modified since the last run of this script, resuming
    # at the last document committed if a previous run did not get to the end
    clip_version = int(index.get("clip_version", default=0))
    last_id = int(index.get("clip_last_id", default=0))

    where = f"version > {clip_version} AND document.id > {last_id}" \
            f" AND mime IN (SELECT id FROM mime WHERE name LIKE 'image/%' OR name LIKE 'video/%')"
    total = index.document_count(where)
    done = 0
    batch = []

    def flush():
        nonlocal done, batch

        if not batch:
            return

        try:
            process_batch(index, batch, model, tag_embeddings, tags, num_tags, color)
        except Exception as e:
            print(f"Could not generate embeddings for {len(batch)} document(s): {e}", file=stderr)

        done += len(batch)
        batch = []
        print_progress(done=done, count=total)

    for doc in index.document_iter(where):

        try:
            batch.append((doc, load_image(index, doc, preprocess)))
        except Exception as e:
            print(f"Could not load image {doc.rel_path}: {e}", file=stderr)
            continue

        if len(batch) >= batch_size:
            flush()

    flush()

    index.set("clip_version", index.versions[-1].id)
    index.set("clip_last_id", 0)

    print("Syncing tag table")
    index.sync_tag_table()
    index.commit()

    print("Done!")


if __name__ == "__main__":
    typer.run(main)
