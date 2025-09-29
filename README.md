# TTS Evaluation Pipeline

This project evaluates Text-to-Speech (TTS) models using ground-truth and generated audio, computing metrics such as MOS, WER, and MCD.

## Requirements

- Python 3.8+
- See `requirements.txt` for required packages.

## Setup

1. **Clone this repository or copy the files to your working directory.**
2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Set your ElevenLabs API key:**
   ```bash
   export ELEVEN_API_KEY=your_elevenlabs_api_key
   ```

## Usage

### 1. Using the LJ Speech Dataset (default)

By default, the script downloads and processes samples from the LJ Speech dataset. It saves ground-truth `.mp3` and `.txt` files in `ground_truth_audios_mp3/`.

Run:

```bash
python eval.py
```

### 2. Using Your Own Files

- To use your own ground-truth `.mp3` and `.txt` files, set `use_local_gt_files = True` in `eval.py`.
- Place your `.mp3` and corresponding `.txt` files in the `ground_truth_audios_mp3/` directory, with matching filenames (e.g., `gt_sample_0.mp3` and `gt_sample_0.txt`).

### 3. Using a Manifest

- Alternatively, provide a `manifest.csv` with columns `audio_filename` and `text`, and place your audio files in `custom_ground_truth_audios/`.

## Output

- Generated audios are saved in `generated_audios_mp3/`.
- Evaluation results are saved to `evaluation_results.csv`.
- A summary is printed to the console.

## Notes

- The script uses OpenAI Whisper for ASR and UTMOS for MOS prediction.
- GPU is used if available.

## How it works

1. Loades the dataset from hugging face (ground truth audio and text)
2. Uses elevenlabs to generate speech from the ground truth text
3. Evalutes the speech on multiple metrics (F1 Score)
