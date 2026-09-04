"""
Generate an LLM-labeled VQA dataset for SUNRGBD scenes using the Deepseek API.

This is a separate question source from the rule-based scripts
(color_questions.py, count_questions.py, object_identification.py,
ProximityQuestion.py, Yes_No_Questions.py) and writes to its own CSV file.
It is never merged into final_dataset.csv.

Deepseek's chat models do not accept image input, so this script grounds
the LLM in the same structured scene annotations (object names + polygon
positions from annotation/index.json) that the rule-based scripts use,
and asks the model to phrase natural-language Q&A pairs from that context.
Color questions are intentionally excluded here since color requires
reading actual pixel values, which a text-only model cannot do.
"""
import argparse
import json
import os
import time

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

from utils import read_paths

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")
OUTPUT_CSV = os.path.join(
    DATASET_DIR, "SUNRGBD_Dataset", "SUNRGBD", "csv_data", "deepseek_dataset.csv"
)
OUTPUT_COLUMNS = ["Questions", "Answers", "Image_Path", "Depth_Path", "Question_Type"]

QUESTIONS_PER_SCENE = 5

SYSTEM_PROMPT = (
    "You write short visual-question-answering pairs for an indoor RGB-D scene. "
    "You are given only the scene's object list and each object's 2D outline "
    "(pixel polygon), not the image itself. Produce natural, varied questions "
    "a person might ask while looking at the room. Cover a mix of these types: "
    "object_identification, counting, proximity, existence. Do not ask about "
    "color, material, or exact pixel coordinates, since those cannot be "
    "verified from the given data. Answers must be short (a word or a few "
    "words) and must be directly supported by the object list provided."
)

RESPONSE_FORMAT_INSTRUCTIONS = (
    'Respond with ONLY a JSON array, no prose, no markdown fences. Each item: '
    '{"question": str, "answer": str, "question_type": one of '
    '["object_identification", "counting", "proximity", "existence"]}. '
    f"Produce exactly {QUESTIONS_PER_SCENE} items."
)


def describe_scene(annotation_data: dict) -> str:
    object_names = [
        obj.get("name", "unknown")
        for obj in annotation_data.get("objects", [])
        if isinstance(obj, dict)
    ]

    polygons_by_object = {}
    for frame in annotation_data.get("frames", []):
        for polygon in frame.get("polygon", []):
            object_index = polygon.get("object")
            xs, ys = polygon.get("x", []), polygon.get("y", [])
            if object_index is None or not xs or not ys:
                continue
            center_x = sum(xs) / len(xs)
            center_y = sum(ys) / len(ys)
            polygons_by_object.setdefault(object_index, []).append((center_x, center_y))

    lines = []
    for index, name in enumerate(object_names):
        centers = polygons_by_object.get(index)
        if centers:
            average_x = sum(c[0] for c in centers) / len(centers)
            average_y = sum(c[1] for c in centers) / len(centers)
            lines.append(f"- {name} (pixel center ~ x={average_x:.0f}, y={average_y:.0f})")
        else:
            lines.append(f"- {name}")

    return "\n".join(lines) if lines else "(no annotated objects)"


def request_questions_for_scene(client: OpenAI, model: str, scene_description: str) -> list[dict]:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Objects in this scene:\n{scene_description}\n\n"
                    f"{RESPONSE_FORMAT_INSTRUCTIONS}"
                ),
            },
        ],
        temperature=0.7,
    )
    content = response.choices[0].message.content.strip()
    return json.loads(content)


def append_rows_to_csv(rows: list[dict], output_path: str) -> None:
    dataframe = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    write_header = not os.path.exists(output_path)
    dataframe.to_csv(output_path, mode="a", header=write_header, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Label SUNRGBD scenes with Deepseek-generated Q&A pairs."
    )
    parser.add_argument(
        "--max_scenes", type=int, default=None, help="Process at most this many scenes."
    )
    parser.add_argument(
        "--model", type=str, default="deepseek-chat", help="Deepseek model name."
    )
    parser.add_argument(
        "--retries", type=int, default=3, help="Retries per scene on API/parse failure."
    )
    args = parser.parse_args()

    load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("Error: DEEPSEEK_API_KEY is missing from your .env file!")
        return

    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    image_paths = read_paths(os.path.join(DATASET_DIR, "all_rgb.txt"))
    depth_paths = read_paths(os.path.join(DATASET_DIR, "all_depth.txt"))
    annotation_paths = read_paths(os.path.join(DATASET_DIR, "annotations.txt"))

    scenes = list(zip(image_paths, depth_paths, annotation_paths))
    if args.max_scenes:
        scenes = scenes[: args.max_scenes]

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    print(f"Writing to {OUTPUT_CSV}")

    for image_path, depth_path, annotation_path in tqdm(scenes, desc="Labeling scenes"):
        full_annotation_path = os.path.join(DATASET_DIR, annotation_path)
        if not os.path.exists(full_annotation_path):
            continue

        with open(full_annotation_path, "r") as annotation_file:
            annotation_data = json.load(annotation_file)

        scene_description = describe_scene(annotation_data)

        for attempt in range(args.retries):
            try:
                qa_items = request_questions_for_scene(client, args.model, scene_description)
                rows = [
                    {
                        "Questions": item["question"],
                        "Answers": item["answer"],
                        "Image_Path": image_path,
                        "Depth_Path": depth_path,
                        "Question_Type": item.get("question_type", "unknown"),
                    }
                    for item in qa_items
                ]
                append_rows_to_csv(rows, OUTPUT_CSV)
                break
            except Exception as error:
                if attempt == args.retries - 1:
                    print(f"\n[Deepseek Error] Giving up on {annotation_path}: {error}")
                else:
                    time.sleep(2**attempt)


if __name__ == "__main__":
    main()
