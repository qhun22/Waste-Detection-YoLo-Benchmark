import os
import glob
import cv2
import yaml
import matplotlib.pyplot as plt

def verify_dataset(yaml_path="data/dataset/data.yaml"):
    if not os.path.exists(yaml_path):
        print(f"Lỗi: Không tìm thấy {yaml_path}")
        return

    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    dataset_dir = os.path.dirname(os.path.abspath(yaml_path))
    names = data.get('names', [])
    print("=" * 40)
    print(f"Danh sách nhãn ({len(names)} classes): {names}")
    print("=" * 40)

    for split in ['train', 'valid', 'test']:
        img_folder = os.path.join(dataset_dir, split, 'images')
        count = len(glob.glob(os.path.join(img_folder, '*.*'))) if os.path.exists(img_folder) else 0
        print(f"Tập {split:<6}: {count:>5} ảnh")

    train_imgs = glob.glob(os.path.join(dataset_dir, 'train', 'images', '*.*'))
    if not train_imgs:
        print("Chưa tìm thấy ảnh trong tập train.")
        return

    sample_img_path = train_imgs[0]
    sample_lbl_path = sample_img_path.replace('images', 'labels').rsplit('.', 1)[0] + '.txt'

    img = cv2.imread(sample_img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = img.shape

    if os.path.exists(sample_lbl_path):
        with open(sample_lbl_path, 'r') as f:
            for line in f.readlines():
                cls_id, x_c, y_c, bw, bh = map(float, line.strip().split())
                
                xmin = int((x_c - bw / 2) * w)
                ymin = int((y_c - bh / 2) * h)
                xmax = int((x_c + bw / 2) * w)
                ymax = int((y_c + bh / 2) * h)

                cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                cls_name = names[int(cls_id)] if int(cls_id) < len(names) else str(cls_id)
                cv2.putText(img, cls_name, (xmin, max(ymin - 5, 15)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    os.makedirs("results/figures", exist_ok=True)
    out_fig = "results/figures/dataset_sample.png"
    plt.figure(figsize=(8, 8))
    plt.imshow(img)
    plt.axis('off')
    plt.title("Sample Bounding Box Verification")
    plt.savefig(out_fig, bbox_inches='tight')
    plt.close()
    print(f"\n=> Đã lưu ảnh kiểm tra nhãn tại: {out_fig}")

if __name__ == "__main__":
    verify_dataset()