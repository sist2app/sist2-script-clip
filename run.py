from io import BytesIO
from sys import stderr

import typer
import clip
import torch
from PIL import Image
from torch import nn

from sist2 import Sist2Index, serialize_float_array, print_progress

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using compute device {DEVICE}")


def load_tag_embeddings(tag_file, model):
    with open(tag_file) as f:
        tags = [line.strip() for line in f]

    text_tokenized = clip.tokenize(tags).to(DEVICE)
    with torch.no_grad():
        tag_embeddings = model.encode_text(text_tokenized)

    print(f"Pre-computed embeddings for {len(tags)} tags")

    return tag_embeddings, tags


def main(index_file, clip_model: str = "ViT-B/32", tags_file: str = "general.txt", num_tags: int = 1, color="#dcd7ff"):
    model, preprocess = clip.load(clip_model, device=DEVICE)
    cosine_sim = nn.CosineSimilarity()

    tag_embeddings, tags = load_tag_embeddings(tags_file, model)

    index = Sist2Index(index_file)

    # Only consider documents that were modified since the last run of this script
    clip_version = index.get("clip_version", default=0)

    index.register_model(
        id=1,
        name="CLIP",
        url="https://raw.githubusercontent.com/sist2app/sist2-models/main/clip/models/clip-vit-base-patch32-q8.onnx",
        path="idx_512.clip",
        size=512,
        type="flat"
    )

    where = f"version > {clip_version} AND ((SELECT name FROM mime WHERE id=document.mime) LIKE 'image/%' OR " \
            f"(SELECT name FROM mime WHERE id=document.mime) LIKE 'video/%')"
    total = index.document_count(where)
    done = 0

    for doc in index.document_iter(where):

        try:
            if doc.parent or doc.mime.startswith("video/"):

                tn = index.get_thumbnail(doc.id)
                if not tn:
                    raise Exception("Could not find thumbnail")

                image = Image.open(BytesIO(tn))
            else:
                image = Image.open(doc.path)
            image = preprocess(image).unsqueeze(0).to(DEVICE)
        except Exception as e:
            print(f"Could not load image {doc.rel_path}: {e}", file=stderr)
            continue

        with torch.no_grad():
            embeddings = model.encode_image(image)

        if num_tags > 0:
            tags_cos_sim = cosine_sim(tag_embeddings, embeddings).cpu().detach().numpy()
            top_n = reversed(tags_cos_sim.argsort()[-num_tags:])
            top_n_tags = [f"clip.{tags[i]}.{color}" for i in top_n]

            if "tag" not in doc.json_data:
                doc.json_data["tag"] = top_n_tags
            else:
                doc.json_data["tag"] = [
                    *(t for t in doc.json_data["tag"] if not t.startswith("clip.")),
                    *top_n_tags
                ]

            index.update_document(doc)

        encoded = serialize_float_array(embeddings.cpu().detach().numpy()[0])

        index.upsert_embedding(doc.id, 0, None, 1, encoded)

        print(
            f"Generated embeddings for {doc.rel_path}"
        )
        done += 1
        print_progress(done=done, count=total)

    index.set("clip_version", index.versions[-1].id)

    print("Syncing tag table")
    index.sync_tag_table()
    index.commit()

    print("Done!")


if __name__ == "__main__":
    typer.run(main)
