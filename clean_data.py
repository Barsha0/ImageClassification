import os

def clean_dataset(folder_path="PetImages"):
    num_skipped = 0

    for subdir in ("Cat", "Dog"):
        full_dir = os.path.join(folder_path, subdir)
        for fname in os.listdir(full_dir):
            fpath = os.path.join(full_dir, fname)
            try:
                with open(fpath, "rb") as f:
                    is_jfif = b"JFIF" in f.peek(10)
                if not is_jfif:
                    num_skipped += 1
                    os.remove(fpath)
            except Exception:
                num_skipped += 1
                os.remove(fpath)

    print(f"Deleted {num_skipped} corrupted images")