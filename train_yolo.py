"""Train a YOLO fire/smoke model on the dataset built by dataset_tools.py.

Train on a laptop with a GPU or on Google Colab, NOT on the Raspberry Pi.
Then copy the resulting best.pt to the Pi and run:
  python fire_detector.py 0 --detector yolo --model best.pt

  pip install ultralytics
  python train_yolo.py dataset/data.yaml --epochs 80
"""
import argparse
import sys

try:
    from ultralytics import YOLO
except ImportError:
    sys.exit("Needs:  pip install ultralytics")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("data", help="path to data.yaml from dataset_tools.py")
    p.add_argument("--model", default="yolov8n.pt", help="start from this pretrained model (n = nano, fast on a Pi)")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    args = p.parse_args()

    model = YOLO(args.model)                 # pretrained on everyday objects, we fine-tune it
    model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                project="runs", name="fire")
    metrics = model.val()
    print(f"mAP50: {metrics.box.map50:.3f}   (how well boxes match your labels, 1.0 is perfect)")
    print("best weights:", getattr(model.trainer, "best", "see the runs/ folder"))


if __name__ == "__main__":
    main()
