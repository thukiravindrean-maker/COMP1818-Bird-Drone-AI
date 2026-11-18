from pathlib import Path
from collections import Counter

CLASSES = {
    0: "airplane",
    1: "bird",
    2: "drone",
    3: "helicopter"
}

# Update these paths to your AOD-4 label folders.
TRAIN_LABELS = Path(
    r"C:\Users\User\Downloads\AOD 4\AOD 4\Annotations\YOLOv8 format\train\labels"
)

TEST_LABELS = Path(
    r"C:\Users\User\OneDrive\Documents\COMP1818_Bird_Drone_Project_Fixed\comp1818-bird-drone\datasets\aod4_evaluation\labels\test"
)

def count_classes(folder):
    counts = Counter()
    files = list(folder.rglob("*.txt"))

    if not folder.is_dir() or not files:
        raise FileNotFoundError(
            f"No YOLO label files found in: {folder}"
        )

    for file in files:
        for line in file.read_text(
            encoding="utf-8"
        ).splitlines():

            parts = line.strip().split()

            if len(parts) >= 5:
                class_id = int(parts[0])

                if class_id not in CLASSES:
                    raise ValueError(
                        f"Unexpected class {class_id} in {file}"
                    )

                counts[class_id] += 1

    return counts


train_counts = count_classes(TRAIN_LABELS)
test_counts = count_classes(TEST_LABELS)

majority_class = max(train_counts, key=train_counts.get)

total_test = sum(test_counts.values())

majority_accuracy = (
    test_counts[majority_class] / total_test
)

random_accuracy = 1 / len(CLASSES)

print("\nAOD-4 BASELINE RESULTS")
print("----------------------")

print("Training class counts:", dict(train_counts))
print("Test class counts:", dict(test_counts))

print(
    "Majority training class:",
    CLASSES[majority_class]
)

print(
    f"Majority-class accuracy: "
    f"{majority_accuracy * 100:.2f}%"
)

print(
    f"Uniform random expected accuracy: "
    f"{random_accuracy * 100:.2f}%"
)

print("\nNote: Classification-only baselines.")
print("Ground-truth object locations are assumed.")
print("These scores are NOT detection mAP.")