# Smart Parking App (Interactive + Billing + History)

This project now includes a complete interactive UI for smart parking flow:

- Permanent member and temporary user selection
- Username/password login for existing permanent members
- Sign up for new permanent members
- Suggested free parking slot preview from CCTV video frame
- Auto timer start when slot is confirmed
- Bill generation when user exits the slot
- QR payment flow for temporary users
- Past bills/history view for permanent members
- **ML-based slot occupancy classification** using a trained CNN

## Tech Stack

- `streamlit` for UI
- `sqlite` for user/session/billing database
- `opencv-python` + `ultralytics` for YOLO car detection
- `torch` + `torchvision` for CNN slot occupancy classifier (PKLot-trained)

## Run

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. (Optional but recommended) put these files in project root:
   - `easy.mp4` (your CCTV parking video)
   - `yolov8s.pt` (YOLO model weights)

3. Start the app:

```bash
streamlit run app.py
```

## Main Files

- `app.py`: UI, auth, flows, timer, billing, history
- `db.py`: SQLite schema + auth/session/history operations
- `parking_logic.py`: free-slot suggestion + preview image (**uses CNN classifier**)
- `detector.py`: original detector script (kept as-is)
- `model.py`: CNN architecture (SlotClassifier)
- `slot_classifier_inference.py`: inference wrapper for the trained CNN
- `prepare_dataset.py`: dataset preparation script for PKLot images
- `train.py`: training script for the SlotClassifier

## Notes

- If `yolov8s.pt` is not available, app still works in demo suggestion mode.
- If `models/slot_classifier.pth` is not found, the app defaults to marking all slots as "Free".
- Database file is created automatically as `parking_app.db`.

---

## ML Model Training

This section explains how to train the CNN parking-slot occupancy classifier used by the app.

### 1. Download the PKLot Dataset

Download the **PKLot (Parking Lot) dataset** from Kaggle:

> **https://www.kaggle.com/datasets/ammarnassanalhajali/pklot-dataset**

Extract the dataset so the folder structure looks like:

```
./data/PKLot/
    Occupied/   ← images of occupied parking slots
    Empty/      ← images of free/empty parking slots
```

### 2. Prepare the Dataset

Run the preparation script to resize all images to 64×64 and create train/val splits:

```bash
python prepare_dataset.py
```

This creates:
- `./data/train.csv` — 80% of images (training set)
- `./data/val.csv` — 20% of images (validation set)
- `./data/PKLot_resized/` — all resized images

### 3. Train the Model

```bash
python train.py
```

Training runs for **15 epochs** and saves the best model (by validation accuracy) to:

```
./models/slot_classifier.pth
```

You should see train loss and validation accuracy printed each epoch in the console.

### 4. Using the Trained Model

Once `slot_classifier.pth` exists in the `./models/` directory, the app **automatically uses it**. No configuration changes are needed — `parking_logic.py` imports the inference module which loads the model on first use.

- **YOLO** is still used for car bounding-box detection (drawing overlays on the video frame)
- **The CNN classifier** replaces the old IoU-based rule logic for deciding whether each slot is occupied or free
- If the model file is missing, the app gracefully falls back to marking all slots as "Free"
