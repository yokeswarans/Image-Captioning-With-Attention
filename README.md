# Image Captioning with Attention

An image captioning project built with an EfficientNet-B4 image encoder, Bahdanau attention, and an LSTM decoder. It can train on MS COCO, caption an image from the command line, or run an interactive Gradio app with optional attention visualizations.

## Quickstart: run the pretrained project

You can try the pretrained model without downloading COCO or training. These steps are for Windows PowerShell.

### 1. Clone the code

```powershell
git clone https://github.com/yokeswarans/Image-Captioning-With-Attention.git
cd Image-Captioning-With-Attention
```

### 2. Install Python dependencies

Python 3.10 or later is recommended. Create a virtual environment and install the project requirements:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks environment activation, run the commands using `.\.venv\Scripts\python.exe` instead of `python`. For CUDA, install the PyTorch and torchvision builds that match your system by following the [PyTorch installation selector](https://pytorch.org/get-started/locally/).

### 3. Download the pretrained weights from Hugging Face

The checkpoint is hosted separately from the source code because it is larger than GitHub's per-file limit. Download it into the `checkpoints` directory:

```powershell
New-Item -ItemType Directory -Force checkpoints | Out-Null
curl.exe -L "https://huggingface.co/Yoki051202/Image-captioning-weights/resolve/main/best_model.pth?download=true" -o "checkpoints\best_model.pth"
```

The downloaded file should be named `best_model.pth` and be approximately 424 MiB. Check that the download completed before continuing:

```powershell
Get-Item checkpoints\best_model.pth | Select-Object Name, Length
```

### 4. Launch the web app

```powershell
python app.py
```

Open the local URL printed in the terminal (normally <http://127.0.0.1:7860>). Upload an image or provide an image URL, select a beam width, and press **Generate caption**. Attention overlays and videos are optional. The app loads `checkpoints\best_model.pth` automatically.

To try the command-line interface instead, replace `path\to\your-image.jpg` with an image file:

```powershell
python predict.py --image "path\to\your-image.jpg"
python predict.py --image "path\to\your-image.jpg" --beam_size 5 --visualize
```

The caption is printed in the terminal. With `--visualize`, attention and caption images are saved under `outputs\predictions\`.

## Features

- EfficientNet-B4 encoder producing a 7 × 7 spatial feature grid.
- Bahdanau attention and an LSTMCell decoder.
- Beam-search caption generation and optional per-word attention visualizations.
- Gradio web app for uploaded images or image URLs, including an optional attention video.
- Training history, metric plots, checkpoints, and a leakage-checked Karpathy-split evaluation.

## Requirements

- Python 3.10 or later.
- PyTorch 2.0+ and a matching `torchvision` installation. A CUDA-capable GPU is recommended for training; CPU is supported but will be considerably slower.
- Internet access for downloading COCO data, ImageNet encoder weights on the first training run, and the clean evaluation's missing held-out images.
- Java on `PATH` for the official `pycocoevalcap` evaluation notebook.

From the project root, create and activate a virtual environment, then install the dependencies:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, use `.\.venv\Scripts\python.exe` in place of `python` in the commands below. For CUDA, install a PyTorch/torchvision build that matches your CUDA setup using the [official PyTorch installation selector](https://pytorch.org/get-started/locally/).

## Dataset

Training expects MS COCO 2017 images and caption annotations in this layout:

```text
data/
└── coco/
    ├── train2017/
    ├── val2017/
    └── annotations/
        ├── captions_train2017.json
        └── captions_val2017.json
```

Download and extract the official [COCO 2017 train images](https://images.cocodataset.org/zips/train2017.zip), [validation images](https://images.cocodataset.org/zips/val2017.zip), and [2017 annotations](https://images.cocodataset.org/annotations/annotations_trainval2017.zip) into `data\coco`. For example, from PowerShell:

```powershell
New-Item -ItemType Directory -Force data\coco | Out-Null
curl.exe -L https://images.cocodataset.org/zips/train2017.zip -o data\coco\train2017.zip
curl.exe -L https://images.cocodataset.org/zips/val2017.zip -o data\coco\val2017.zip
curl.exe -L https://images.cocodataset.org/annotations/annotations_trainval2017.zip -o data\coco\annotations_trainval2017.zip
Expand-Archive data\coco\train2017.zip -DestinationPath data\coco
Expand-Archive data\coco\val2017.zip -DestinationPath data\coco
Expand-Archive data\coco\annotations_trainval2017.zip -DestinationPath data\coco
```

The archives are large. The working copy may contain only a subset of the train images; make sure the complete `train2017` and `val2017` image directories are extracted before expecting full-dataset training/evaluation. `instances_*.json` and keypoint annotations are not used by this captioning pipeline.

## Train

After installing dependencies and placing the COCO files in the layout above, open PowerShell in the project root and start training:

```powershell
python train.py --epochs 50 --batch_size 32 --patience 15 --num_workers 0
```

`--num_workers 0` is a safe starting point on Windows. Increase it if your setup benefits from parallel data loading. Defaults are 50 epochs, batch size 32, early stopping patience 15, automatic CUDA/CPU selection, and mixed precision when CUDA is available. Training starts from ImageNet-pretrained EfficientNet-B4 weights. To resume an interrupted run from `checkpoints/last_model.pth`:

```powershell
python train.py --epochs 50 --patience 15 --batch_size 16 --num_workers 0 --resume
```

At startup, training checks image IDs in the train and validation annotations and removes any overlapping IDs from validation before creating its data loader. Exclusions are reported in `runs/train.log`; a clean split reports that the IDs are disjoint. This prevents an image used for training from also contributing to validation loss, BLEU, METEOR, ROUGE-L, or CIDEr-D. With the supplied COCO 2017 annotation files, the split contains 118,287 training IDs and 5,000 validation IDs, with no overlap.

To watch progress, read the live terminal output or open another PowerShell window in the project root and run:

```powershell
Get-Content runs\train.log -Wait
```

Training writes per-epoch metrics to `runs/history.json`, updates metric/loss plots in `runs/`, and saves the best and latest checkpoints in `checkpoints/`. The validation metrics are computed on at most `--val_batches` batches (50 by default), rather than a full COCO evaluation. Useful options include `--device cuda` or `--device cpu`, `--no_amp`, and `--val_batches 50`. See `python train.py --help` for all options. The `run_train.sh` helper is an optional Bash/tmux launcher for Linux or WSL; the Python command above works directly on Windows.

Training creates:

- `checkpoints/best_model.pth` and `checkpoints/last_model.pth`
- `runs/history.json` with per-epoch train/validation loss and validation metrics
- `runs/train.log`, `runs/STATUS`, and `runs/status.json`
- `runs/train_val_loss.png`, `runs/bleu_scores.png`, `runs/other_metrics.png`, and `runs/schedule.png`

Validation metrics during training are computed on at most `--val_batches` validation batches (50 by default), rather than a full COCO evaluation.

## Generate a caption

With a trained checkpoint at the default path:

```powershell
python predict.py --image test.webp
```

Choose a beam width and optionally save a caption figure and attention map:

```powershell
python predict.py --image test.webp --beam_size 5 --visualize
```

The generated caption is printed in the terminal. With `--visualize`, PNGs are written to `outputs/predictions/`. Use `--checkpoint PATH` to select another checkpoint or `--device cpu` / `--device cuda` to override automatic device selection.

## Run the web app

```powershell
python app.py
```

Open the local URL printed by Gradio (by default `http://127.0.0.1:7860`). The app automatically looks for `checkpoints/best_model.pth`, then `checkpoints/last_model.pth`; if neither exists, train the model or pass a checkpoint explicitly:

```powershell
python app.py --checkpoint checkpoints\best_model.pth --host 127.0.0.1 --port 7860
```

The UI accepts an uploaded image or image URL, supports beam widths 5 and 7, and can show attention overlays or an attention video. The optional `--share` flag creates a public Gradio link; only use it when you intend to expose the running app.

## View metrics and evaluate

### Training metrics

After training, inspect `runs/history.json` for the recorded numbers and open the plots:

- `runs/train_val_loss.png`
- `runs/bleu_scores.png`
- `runs/other_metrics.png`
- `runs/schedule.png`

The web app also shows a short training readout from `runs/history.json` when that file exists.

### Leakage-checked Karpathy test evaluation

For a more complete, official COCO-caption evaluation, open `evaluate_karpathy_clean.ipynb` in VS Code or Jupyter and run its cells from the project root. The notebook uses beam search and `pycocoevalcap`; it checks for overlap with the training annotations and excludes contaminated images. It uses available local COCO images and downloads missing clean held-out images when needed. Install the notebook/evaluation extras if they are not already available:

```powershell
python -m pip install jupyter tqdm
java -version
```

Install a Java runtime and make sure `java` is on `PATH` if the Java check fails. Results are saved as timestamped JSON files under `outputs/eval_karpathy_clean/`. The older `evaluate_karpathy.ipynb` uses a contaminated image set and should not be used to report generalization scores.

### Existing clean evaluation snapshot

The repository currently contains an evaluation snapshot using beam size 5. It evaluated 593 clean images from the 5,000-image Karpathy test split after excluding 4,407 images identified as overlapping with training annotations. These scores are a reference for that saved checkpoint and sample, not a guarantee that a fresh training run will reproduce them:

| Metric | Score |
| --- | ---: |
| BLEU-1 | 0.7488 |
| BLEU-2 | 0.5893 |
| BLEU-3 | 0.4564 |
| BLEU-4 | 0.3541 |
| METEOR | 0.2647 |
| ROUGE-L | 0.5613 |
| CIDEr | 1.1288 |

Source: `outputs/eval_karpathy_clean/karpathy_eval_clean_20260809_222556.json`. Since this clean subset is smaller than the full test set, interpret its scores with that limitation in mind.

## Project layout

```text
app.py                         Gradio web app
predict.py                     Single-image inference CLI
train.py                       Training entry point
run_train.sh                   Optional Linux/WSL background-training helper
captioning/                    Dataset, model, training utilities, metrics, inference
webui/theme.css                Web app styling
evaluate_karpathy_clean.ipynb  Leakage-checked official evaluation
data/coco/                     COCO images, annotations, and generated caches
checkpoints/                   Saved model checkpoints
runs/                          Logs, history, and metric plots
outputs/                       Predictions and evaluation results
```

## Notes on sharing

COCO images and annotations, generated caches/predictions, and model checkpoints can make a Git repository very large. Check their licenses and repository hosting limits before publishing them. A fresh clone needs the COCO files to train and a compatible checkpoint to run inference immediately; otherwise, train a checkpoint locally first. The code and the clean evaluation notebook are separate from the COCO dataset's redistribution terms.
