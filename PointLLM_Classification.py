import argparse
import os
import pandas as pd
from transformers import AutoTokenizer
import torch
import numpy as np
import open3d as o3d
from pointllm.conversation import conv_templates
from pointllm.utils import disable_torch_init
from pointllm.model import *
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from pointllm.data import pc_norm, farthest_point_sample

def set_seed(seed):
    """
    Set random seed for reproducibility.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def load_point_cloud_from_file(file_path):
    """
    Load a custom point cloud from a .ply file, normalize it, and downsample if necessary.
    """
    print(f"[INFO] Loading point cloud from file: {file_path}")
    pcd = o3d.io.read_point_cloud(file_path)
    points = np.asarray(pcd.points, dtype=np.float32)
    colors = np.asarray(pcd.colors, dtype=np.float32)

    if colors.size == 0:
        colors = np.zeros_like(points, dtype=np.float32)

    colors = np.clip(colors, 0, 1)
    points = np.concatenate((points, colors), axis=1)

    if points.shape[0] > 8192:
        points = farthest_point_sample(points, 8192)

    points = pc_norm(points)
    return torch.from_numpy(points).unsqueeze_(0).to(torch.float32).cuda()

def init_model(args):
    disable_torch_init()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = PointLLMLlamaForCausalLM.from_pretrained(
        args.model_name, low_cpu_mem_usage=False, use_cache=True
    ).cuda()
    model.initialize_tokenizer_point_backbone_config_wo_embedding(tokenizer)
    model.eval()
    conv = conv_templates["vicuna_v1_1"].copy()
    return model, tokenizer, model.get_model().point_backbone_config, conv

def classify_point_cloud(point_cloud, model, tokenizer, conv, context):
    allowed_classes = [
        "decoration", "arch", "column", "roof", "floor", "door/window", "wall", "stairs", "vault"
    ]

    classification_prompt = (
        f"Classify this architectural subpart, coming from a {context}, with one class among: {', '.join(allowed_classes)}.\n\n" 
        "These are the descriptions of the classes, specifically tailored for architectural elements:\n"
        "decoration: Ornamental elements used to enhance the aesthetic appeal of buildings or spaces. Decorations can include sculptures, carvings, frescoes, mosaics, or other artistic details. They are often found on walls, ceilings, or structural elements like columns and arches.\n"
        "arch: A curved structural element that spans an opening and distributes weight above it. Arches are commonly used in doorways, windows, or as part of arcades in cloisters. They can be plain or decorated with intricate carvings.\n"
        "column: A vertical, cylindrical structural element that supports other architectural components, such as arches, roofs, or entablatures. Columns can be plain or adorned with capitals (e.g., Doric, Ionic, Corinthian) and fluting. They are often found in porticos, cloisters, or as standalone decorative features.\n"
        "roof: The upper covering of a building that protects it from the elements. Roofs can be sloped, flat, or domed, and may include decorative elements such as tiles, statues, or spires. In churches, roofs often feature intricate wooden or stone frameworks.\n"
        "floor: The horizontal surface on which people walk. Floors can be made of stone, wood, tiles, or other materials and may include decorative patterns such as mosaics or inlays. In churches, floors often feature symbolic designs or inscriptions.\n"
        "door/window: Openings in walls that allow access, light, or ventilation. Doors are typically made of wood or metal and may include carvings or decorative panels. Windows often feature frames in stone or wood and may include stained glass or tracery in churches.\n"
        "wall: A vertical structural element that encloses or divides spaces. Walls can be plain or decorated with frescoes, reliefs, or other artistic details. In churches, walls often include niches, altars, or inscriptions.\n"
        "stairs: A series of steps that connect different levels of a building. Stairs can be straight, spiral, or curved and are often made of stone, wood, or metal. They may include decorative railings or balustrades.\n"
        "vault: A curved architectural structure that forms the ceiling or roof of a space. Vaults are commonly found in churches and can be barrel-shaped, groin-shaped, or ribbed. They often include decorative elements such as frescoes or sculptural details.\n\n"
        "Instructions:\nYou have to reason on the elements you see, and provide a direct answer as the predicted_class.\n"
        
    )


    conv.reset()
    conv.append_message(conv.roles[0], classification_prompt)
    conv.append_message(conv.roles[1], None)
    full_prompt = conv.get_prompt()

    print(f"[QUERY] {full_prompt}")

    input_ids = torch.as_tensor(tokenizer([full_prompt]).input_ids).cuda()

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            point_clouds=point_cloud,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            max_new_tokens=100,
            top_p=0.9,
        )

    response = tokenizer.batch_decode(output_ids[:, input_ids.shape[1]:], skip_special_tokens=True)[0].strip()
    print(f"[RESPONSE] {response}")

    classification = response.lower()

    # Handle cases where "door" or "window" are mentioned
    if "door" in classification or "window" in classification:
        return "Door/Window", response

    # Match other allowed classes
    for cls in allowed_classes:
        if cls.lower() in classification:
            return cls.capitalize(), response

    print(f"[INFO] Most Related Class: {response}")
    return "Unknown", response


def calculate_metrics(actual_classes, predicted_classes, calculation_file, class_labels):
    normalized_class_labels = [label.lower() for label in class_labels]
    actual_classes = [actual.lower() for actual in actual_classes]
    predicted_classes = [pred.lower() for pred in predicted_classes]

    valid_indices = [i for i, pred in enumerate(predicted_classes) if pred != "unknown"]
    filtered_actual = [actual_classes[i] for i in valid_indices]
    filtered_predicted = [predicted_classes[i] for i in valid_indices]

    filtered_actual = [label for label in filtered_actual if label in normalized_class_labels]
    filtered_predicted = [pred for pred in filtered_predicted if pred in normalized_class_labels]

    if not filtered_actual or not filtered_predicted:
        print("[ERROR] No valid labels matching allowed classes after alignment.")
        calculation_file.write("\n[ERROR] No valid labels matching allowed classes after alignment.\n")
        return

    accuracy = accuracy_score(filtered_actual, filtered_predicted)
    precision = precision_score(filtered_actual, filtered_predicted, average='weighted', zero_division=0)
    recall = recall_score(filtered_actual, filtered_predicted, average='weighted', zero_division=0)
    f1 = f1_score(filtered_actual, filtered_predicted, average='weighted', zero_division=0)

    calculation_file.write("\n[INFO] Performance Metrics:\n")
    calculation_file.write(f"Accuracy: {accuracy:.2f}\n")
    calculation_file.write(f"Precision: {precision:.2f}\n")
    calculation_file.write(f"Recall: {recall:.2f}\n")
    calculation_file.write(f"F1 Score: {f1:.2f}\n")

    print("\n[INFO] Performance Metrics:")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")

    try:
        cm = confusion_matrix(filtered_actual, filtered_predicted, labels=normalized_class_labels)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=normalized_class_labels)
        disp.plot(cmap='viridis', xticks_rotation='vertical')
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.savefig("confusion_matrix.png")
        print("[INFO] Confusion matrix saved as 'confusion_matrix.png'")
    except ValueError as e:
        print(f"[ERROR] Unable to generate confusion matrix: {e}")
        calculation_file.write(f"\n[ERROR] Unable to generate confusion matrix: {e}\n")

def classify_point_clouds(args, model, tokenizer, point_backbone_config, conv, csv_path, log_file, calculation_file):
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"[ERROR] Failed to read CSV file: {e}")
        return

    df = df.dropna(subset=['NAME', 'CLASS', 'PLACE'])
    df['NAME'] = df['NAME'].astype(str)
    df['CLASS'] = df['CLASS'].astype(str)
    df['PLACE'] = df['PLACE'].astype(str)

    allowed_classes = [
        "decoration", "arch", "column", "roof", "floor", "door/window", "wall", "stairs", "vault"
    ]
    predicted_classes = []
    actual_classes = []

    for _, row in df.iterrows():
        file_name, actual_class, place = row['NAME'], row['CLASS'], row['PLACE']
        file_path = os.path.join(args.folder_path, f"{file_name}.ply")

        if not os.path.exists(file_path):
            print(f"[WARNING] File not found: {file_path}. Skipping.")
            continue

        try:
            point_cloud = load_point_cloud_from_file(file_path)
        except Exception as e:
            print(f"[ERROR] Could not load point cloud: {e}. Skipping.")
            continue

        context = (
            "cloister" if "cloister" in file_name.lower()
            else "church" if "church" in file_name.lower()
            else "Courtroom of a Castle" if "room" in file_name.lower()
            else "Chapel" if "chapel" in file_name.lower()
            else "Pavilion" if "pavillion" in file_name.lower()
            else "Portico" if "portico" in file_name.lower()
            else "building"
        )

        predicted_class, response = classify_point_cloud(point_cloud, model, tokenizer, conv, context)
        predicted_classes.append(predicted_class)
        actual_classes.append(actual_class)

        log_file.write(f"File: {file_name}, Actual: {actual_class}, Predicted: {predicted_class}, Place: {place}, Response: {response}\n")

    

    # Calculate overall metrics
    calculate_metrics(actual_classes, predicted_classes, calculation_file, allowed_classes)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="RunsenXu/PointLLM_7B_v1.2")
    parser.add_argument("--folder_path", type=str, required=True)
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--log_file", type=str, default="classification_log.txt")
    parser.add_argument("--calculation_file", type=str, default="calculation.txt")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    with open(args.log_file, "w") as log_file, open(args.calculation_file, "w") as calculation_file:
        model, tokenizer, point_backbone_config, conv = init_model(args)
        classify_point_clouds(args, model, tokenizer, point_backbone_config, conv, args.csv_path, log_file, calculation_file)