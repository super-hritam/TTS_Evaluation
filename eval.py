import os
import torch
import torchaudio
import librosa
import numpy as np
from dtw import dtw
from datasets import load_dataset
from transformers import pipeline
import pandas as pd
from jiwer import wer
import logging
from elevenlabs import ElevenLabs, VoiceSettings

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("tts_evaluation")

# Initialize ElevenLabs client with environment variable
elevenlabs_client = ElevenLabs(
    api_key=os.environ.get("ELEVEN_API_KEY", "<API_KEY>"),
)

def elevenlabs_tts(text, voice_id="bIHbv24MWmeRgasZH58o", model_id="eleven_turbo_v2_5",
                  voice_settings=VoiceSettings(stability=0.5, similarity_boost=0.5),
                  seed=42, output_format="mp3_22050_32"):
    """Generate streaming TTS audio using ElevenLabs."""
    if not os.environ.get("ELEVEN_API_KEY"):
        raise ValueError("ELEVEN_API_KEY environment variable not set")
    try:
        audio_stream = elevenlabs_client.text_to_speech.stream(
            voice_id=voice_id,
            output_format=output_format,
            text=text,
            model_id=model_id,
            voice_settings=voice_settings,
            seed=seed
        )
        return audio_stream
    except Exception as e:
        logger.error(f"Error generating TTS audio: {e}")
        raise

def compute_mcd(gt_path, gen_path, n_mfcc=13, sr=22050):
    """Compute Mel Cepstral Distortion using dtw-python from MP3 files."""
    try:
        # librosa can load mp3 files directly
        gt_audio, _ = librosa.load(gt_path, sr=sr)
        gen_audio, _ = librosa.load(gen_path, sr=sr)

        # Compute MFCCs
        gt_mfcc = librosa.feature.mfcc(y=gt_audio, sr=sr, n_mfcc=n_mfcc)
        gen_mfcc = librosa.feature.mfcc(y=gen_audio, sr=sr, n_mfcc=n_mfcc)

        # Compute DTW-aligned distance
        dist, _, _, _ = dtw(gt_mfcc.T, gen_mfcc.T, dist=lambda x, y: np.linalg.norm(x - y))
        mcd = dist / gen_mfcc.shape[1]
        return mcd
    except Exception as e:
        logger.error(f"Error computing MCD for {gt_path}, {gen_path}: {e}")
        return float('inf')

def compute_classification_metrics(gt_text, pred_text):
    """Compute TP, FP, FN, TN, precision, recall, and F1 from WER errors."""
    try:
        error = wer(gt_text, pred_text, standardize=True)
        # In jiwer, hits = S + D + I - error
        # S = Substitutions, D = Deletions, I = Insertions
        # TP (hits) = Total words in reference - S - D
        # So we use the breakdown from jiwer's `wer` function directly.
        measures = wer(gt_text.strip().lower(), pred_text.strip().lower(),
                       truth_transform=None, hypothesis_transform=None,
                       concatenate_texts=False)

        tp = measures.hits
        fp = measures.insertions
        fn = measures.deletions
        tn = 0 # Cannot be computed in this context

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": precision, "recall": recall, "f1": f1}
    except Exception as e:
        logger.error(f"Error computing classification metrics: {e}")
        return {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "precision": 0, "recall": 0, "f1": 0}


def main():
    # Step 1: Setup directories and device
    output_dir = "generated_audios_mp3"
    gt_dir = "ground_truth_audios_mp3"
    custom_gt_dir = "custom_ground_truth_audios"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)
    os.makedirs(custom_gt_dir, exist_ok=True) # Ensure custom dir exists
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    num_samples = 10  # Number of samples to evaluate

    # Step 2: Load data - either from Hugging Face dataset or local manifest or local files
    dataset_items = []
    use_local_gt_files = False  # Set this to True to use local .mp3/.txt files instead of loading dataset

    try:
        # --- TO USE YOUR OWN FILES, COMMENT OUT THE LINE BELOW ---
        if not use_local_gt_files:
            dataset = load_dataset("lj_speech", split=f"train[:{num_samples}]")
        else:
            dataset = None
        # ---------------------------------------------------------
    except Exception:
        dataset = None

    if dataset:
        logger.info(f"Loaded dataset with {len(dataset)} samples. Processing...")
        for i, item in enumerate(dataset):
            text = item["text"]
            gt_path = os.path.join(gt_dir, f"gt_sample_{i}.mp3")
            txt_path = os.path.join(gt_dir, f"gt_sample_{i}.txt")
            # Save ground-truth audio as MP3 for consistent comparison
            torchaudio.save(gt_path, torch.tensor(item["audio"]["array"]).unsqueeze(0), 
                            item["audio"]["sampling_rate"], format="mp3")
            # Save ground-truth text as .txt
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(text)
            dataset_items.append({"text": text, "gt_path": gt_path, "id": i})
    else:
        logger.info("Dataset not loaded. Looking for local files in manifest.csv or ground-truth directory...")
        manifest_path = "manifest.csv"
        if os.path.exists(manifest_path):
            manifest = pd.read_csv(manifest_path)
            for index, row in manifest.iterrows():
                text = row["text"]
                audio_filename = row["audio_filename"]
                gt_path = os.path.join(custom_gt_dir, audio_filename)
                if not os.path.exists(gt_path):
                    logger.warning(f"Audio file {gt_path} not found. Skipping.")
                    continue
                dataset_items.append({"text": text, "gt_path": gt_path, "id": index})
        else:
            # Look for .mp3 and .txt pairs in gt_dir
            logger.info(f"{manifest_path} not found. Looking for .mp3/.txt pairs in {gt_dir}...")
            gt_files = [f for f in os.listdir(gt_dir) if f.endswith(".mp3")]
            for gt_file in gt_files:
                base = os.path.splitext(gt_file)[0]
                txt_file = base + ".txt"
                gt_path = os.path.join(gt_dir, gt_file)
                txt_path = os.path.join(gt_dir, txt_file)
                if os.path.exists(txt_path):
                    with open(txt_path, "r", encoding="utf-8") as f:
                        text = f.read().strip()
                    # Extract id from filename if possible
                    try:
                        sample_id = int(base.split("_")[-1])
                    except Exception:
                        sample_id = base
                    dataset_items.append({"text": text, "gt_path": gt_path, "id": sample_id})
                else:
                    logger.warning(f"Text file {txt_path} not found for audio {gt_path}. Skipping.")

    if not dataset_items:
        logger.error("No valid samples to process. Exiting.")
        return

    # Step 3: Generate audios and prepare for metrics
    evaluation_data = []
    for item in dataset_items:
        text = item["text"]
        gt_path = item["gt_path"]
        sample_id = item["id"]
        
        gen_path = os.path.join(output_dir, f"gen_sample_{sample_id}.mp3")

        # Generate audio if it doesn't exist
        if not os.path.exists(gen_path):
            try:
                logger.info(f"Generating audio for sample {sample_id}...")
                audio_stream = elevenlabs_tts(text)
                with open(gen_path, "wb") as f:
                    for chunk in audio_stream:
                        if chunk:
                            f.write(chunk)
            except Exception as e:
                logger.error(f"Failed to generate audio for sample {sample_id}: {e}")
                continue
        
        evaluation_data.append({"text": text, "gt_path": gt_path, "gen_path": gen_path, "id": sample_id})

    # Step 4: Compute metrics
    try:
        whisper = pipeline("automatic-speech-recognition", model="openai/whisper-small", device=device)
        predictor = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True).to(device)
    except Exception as e:
        logger.error(f"Error initializing MOS/Whisper models: {e}")
        return

    all_metrics = []
    for data in evaluation_data:
        gen_path = data["gen_path"]
        gt_path = data["gt_path"]
        gt_text = data["text"]
        
        # Predicted MOS
        try:
            waveform, sr = torchaudio.load(gen_path)
            mos = predictor(waveform.to(device), sr=sr).item()
        except Exception as e:
            logger.error(f"Error computing MOS for {gen_path}: {e}")
            mos = float('nan')

        # WER and Classification Metrics
        try:
            pred_text = whisper(gen_path)["text"]
            wer_score = wer(gt_text, pred_text)
            class_metrics = compute_classification_metrics(gt_text, pred_text)
        except Exception as e:
            logger.error(f"Error computing WER for {gen_path}: {e}")
            wer_score = float('inf')
            class_metrics = {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "precision": 0, "recall": 0, "f1": 0}

        # MCD
        try:
            mcd_score = compute_mcd(gt_path, gen_path)
        except Exception as e:
            logger.error(f"Error computing MCD for {gt_path}, {gen_path}: {e}")
            mcd_score = float('inf')
            
        current_metrics = {
            "mos": mos, "wer": wer_score, "mcd": mcd_score, **class_metrics
        }
        all_metrics.append(current_metrics)

    # Aggregate and save results
    results_df = pd.DataFrame(all_metrics)
    avg_results = results_df.mean()
    sum_results = results_df[["tp", "fp", "fn", "tn"]].sum()

    final_report = pd.DataFrame({
        "Metric": ["MOS", "WER", "MCD",
                   "True Positives", "False Positives", "False Negatives",
                   "Precision (Avg)", "Recall (Avg)", "F1-Score (Avg)"],
        "Value": [avg_results.get("mos"), avg_results.get("wer"), avg_results.get("mcd"),
                  sum_results.get("tp"), sum_results.get("fp"), sum_results.get("fn"),
                  avg_results.get("precision"), avg_results.get("recall"), avg_results.get("f1")]
    })

    csv_path = "evaluation_results.csv"
    try:
        final_report.to_csv(csv_path, index=False)
        logger.info(f"Results saved to {csv_path}")
        print("\n--- Evaluation Summary ---")
        print(final_report)
        print("------------------------")
    except Exception as e:
        logger.error(f"Error saving results: {e}")

if __name__ == "__main__":
    main()