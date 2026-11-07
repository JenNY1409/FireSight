# FireSight

A small drone that looks down on a forest, spots fire with computer vision, and rates
how serious it is. This repo holds the vision code that runs on the drone's Raspberry Pi 4,
the tools to build a training set, and the project website.

**Status: in development.** The detection logic is written and tested on synthetic images.
It has not yet been trained or tested on real fire footage, and has not flown.

## What's here

| File | What it does |
|---|---|
| `fire_detector.py` | Detects fire, scores severity, filters false alarms, logs JSON. Works on an image, a video, or a live USB camera. |
| `dataset_tools.py` | Builds YOLO train/val folders from your labeled photos and sanity-checks them. |
| `train_yolo.py` | Trains a YOLO model on that dataset (run on a laptop or Colab, not the Pi). |
| `test_*.py` | Unit tests. |
| `docs/index.html` | The project website (GitHub Pages). |

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python fire_detector.py forest.jpg                          # one image
python fire_detector.py flight.mp4 --fps 5                  # a video
python fire_detector.py 0 --altitude 40 --hfov 95           # live camera 0
python fire_detector.py 0 --detector yolo --model best.pt   # with trained weights

python -m unittest discover -v                              # run the tests
```

Each analyzed frame prints one JSON line (mode, level, score, fire %, smoke %, estimated
area in m² if you pass `--altitude` and `--hfov`, and whether the alert is confirmed).
Confirmed alerts save an annotated image to `out/`.

## How severity works

```
score = 0.7 * min(fire share of frame / 10%, 1) + 0.3 * brightness of the fire
score < 0.25 LOW   |   < 0.55 MODERATE   |   otherwise HIGH
```

An alert is only confirmed if fire appears in 3 of the last 5 analyzed frames.
The thresholds are starting guesses in `Config` and need tuning on real footage.

## The camera

An Arducam 1080p USB camera with a day/night IR-cut filter: color in daylight, gray in the
dark. It is **not thermal**, so severity comes from how large and bright the fire looks,
not its temperature.

In night mode the frames are gray, so the color detector cannot see fire. Use the YOLO
detector there. The `mode` and `color_sat` fields in the output let you check that the
day/night threshold (`night_sat_thresh`) is right for your camera.

## Training your own model

1. Take photos in day and night modes, one subfolder per session. Include negatives
   (lamps, the sun, empty scenes) with no label file.
2. Label with Roboflow, CVAT or LabelImg and export YOLO format (0 = fire, 1 = smoke).
3. Build and check the dataset, then train:

```bash
python dataset_tools.py split raw dataset --add-gray
python dataset_tools.py check dataset
python train_yolo.py dataset/data.yaml
```

Whole sessions go to train or val, never both, so the validation score is honest.
`--add-gray` adds gray copies of training images to teach night mode.

## Known limits

- The color detector is a baseline. Sunsets, orange roofs and red vehicles can fool it.
- Photos taken close up look different from a forest seen from 40 m. Include distant
  or aerial images in training.
- The area estimate assumes the camera points straight down.

## Safety

Test only with small, controlled, permitted fires. Never fly near a real wildfire; doing so
is illegal in the US and endangers firefighting aircraft.
