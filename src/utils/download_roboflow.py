import os
import shutil
from roboflow import Roboflow

def download_data():

    rf = Roboflow(api_key="jSpRyxkSrSOYaBWQ0dJN")
    project = rf.workspace("material-identification").project("garbage-classification-3")
    version = project.version(2)

    print("Đang tải dataset từ Roboflow...")

    dataset = version.download("yolov8")

    target_dir = os.path.abspath("data/dataset")
    source_dir = os.path.abspath(dataset.location)

    if source_dir != target_dir:
        os.makedirs(target_dir, exist_ok=True)
        for item in os.listdir(source_dir):
            s = os.path.join(source_dir, item)
            d = os.path.join(target_dir, item)
            if os.path.exists(d):
                if os.path.isdir(d):
                    shutil.rmtree(d)
                else:
                    os.remove(d)
            shutil.move(s, d)
        if os.path.exists(source_dir) and source_dir != os.path.abspath("."):
            shutil.rmtree(source_dir, ignore_errors=True)

    print(f"\n=> Tải dữ liệu thành công! Vị trí: {target_dir}")

if __name__ == "__main__":
    download_data()