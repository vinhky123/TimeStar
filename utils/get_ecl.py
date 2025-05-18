import kagglehub
import os
import shutil
from pathlib import Path


def download_kaggle_datasets(
    dataset_path: str, output_dir: str = "./dataset/electricity"
):
    try:
        path = kagglehub.dataset_download(dataset_path)
        print(f"Dataset downloaded to: {path}")

        source_path = Path(path)
        if source_path.exists():
            for item in source_path.iterdir():
                target = Path(output_dir) / item.name
                if target.exists():
                    if target.is_dir():
                        shutil.rmtree(target)
                    else:
                        target.unlink()
                shutil.move(str(item), str(target))

            if not any(source_path.iterdir()):
                source_path.rmdir()

        print(f"Successfully moved all files to {output_dir}")

    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    dataset_path = "levinhkyyy/ecl-dataset"
    download_kaggle_datasets(dataset_path)
