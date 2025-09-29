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
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("tts_evaluation")

# Initialize ElevenLabs client with environment variable
elevenlabs_client = ElevenLabs(
    api_key=os.environ.get("ELEVEN_API_KEY", "<API_KEY>"),
    base_url="https://api-global-preview.elevenlabs.io"
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
    """Compute Mel Cepstral Distortion using dtw-python."""
    try:
        gt_audio, _ = librosa.load(gt_path, sr=sr)
        gen_audio, _ = librosa.load(gen_path, sr=sr)
        
        # Compute MFCCs
        gt_mfcc = librosa.feature.mfcc(y=gt_audio, sr=sr, n_mfcc=n_mfcc)
        gen_mfcc = librosa.feature.mfcc(y=gen_audio, sr=sr, n_mfcc=n_mfcc)
        
        # Ensure same length for DTW
        min_len = min(gt_mfcc.shape[1], gen_mfcc.shape[1])
        gt_mfcc = gt_mfcc[:, :min_len]
        gen_mfcc = gen_mfcc[:, :min_len]
        
        # Compute DTW-aligned distance
        dist, _, _, _ = dtw(gt_mfcc.T, gen_mfcc.T, dist=lambda x, y: np.linalg.norm(x - y))
        mcd = dist / min_len
        return mcd
    except Exception as e:
        logger.error(f"Error computing MCD for {gt_path}, {gen_path}: {e}")
        return float('inf')

def compute_classification_metrics(gt_text, pred_text):
    """Compute TP, FP, FN, TN, precision, recall, and F1 from WER errors."""
    try:
        measures = compute_measures(gt_text.strip().lower(), pred_text.strip().lower())
        tp = measures["hits"]  # Correct words (true positives)
        fp = measures["insertions"]  # Words in pred but not in gt (false positives)
        fn = measures["deletions"]  # Words in gt but not in pred (false negatives)
        tn = 0  # Approximate TN as 0 (minimal non-word context in TTS)
        
        # Compute precision, recall, F1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
    except Exception as e:
        logger.error(f"Error computing classification metrics: {e}")
        return {
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "tn": 0,
            "precision": 0,
            "recall": 0,
            "f1": 0
        }

# Minimal compute_measures implementation for jiwer compatibility
def compute_measures(truth, hypothesis):
    """
    Returns a dict with 'hits', 'insertions', 'deletions' for two strings.
    This is a simplified version for word-level comparison.
    """
    truth_words = truth.strip().split()
    hyp_words = hypothesis.strip().split()

    # Levenshtein distance matrix
    d = np.zeros((len(truth_words)+1, len(hyp_words)+1), dtype=int)
    for i in range(len(truth_words)+1):
        d[i][0] = i
    for j in range(len(hyp_words)+1):
        d[0][j] = j
    for i in range(1, len(truth_words)+1):
        for j in range(1, len(hyp_words)+1):
            if truth_words[i-1] == hyp_words[j-1]:
                cost = 0
            else:
                cost = 1
            d[i][j] = min(
                d[i-1][j] + 1,      # deletion
                d[i][j-1] + 1,      # insertion
                d[i-1][j-1] + cost  # substitution
            )
    # Backtrack to count hits, insertions, deletions
    i, j = len(truth_words), len(hyp_words)
    hits = 0
    insertions = 0
    deletions = 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and truth_words[i-1] == hyp_words[j-1]:
            hits += 1
            i -= 1
            j -= 1
        elif i > 0 and (j == 0 or d[i][j] == d[i-1][j] + 1):
            deletions += 1
            i -= 1
        elif j > 0 and (i == 0 or d[i][j] == d[i][j-1] + 1):
            insertions += 1
            j -= 1
        else:
            i -= 1
            j -= 1
    return {"hits": hits, "insertions": insertions, "deletions": deletions}

def main():
    # Step 1: Setup directories and device
    output_dir = "generated_audios"
    gt_dir = "ground_truth_audios"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    num_samples = 10  # Number of samples to evaluate

    # Step 2: Load dataset (Only run if dataset is not already downloaded)
    try:
        dataset = load_dataset("lj_speech", split=f"train[:{num_samples}]")
        logger.info(f"Loaded dataset with {len(dataset)} samples")
    except Exception as e:
        logger.error(
            f"Error loading dataset: {e}. "
            "Run 'huggingface-cli delete-cache' or manually download LJSpeech from "
            "https://huggingface.co/datasets/lj_speech and load with "
            "load_dataset('/path/to/lj_speech', split='train[:10]')"
        )
        return

    # Step 3: Verify dataset and generate audios
    tsv_data = []
    wav_paths = []
    gt_paths = []
    valid_samples = []
    for i, item in enumerate(dataset):
        text = item["text"]
        audio_path = item.get("audio", {}).get("path")
        if not audio_path or not os.path.exists(audio_path):
            logger.warning(f"Skipping sample {i}: Audio file missing at {audio_path}")
            continue
        
        valid_samples.append((i, item))
        mp3_path = os.path.join(output_dir, f"sample_{i}.mp3")
        wav_path = os.path.join(output_dir, f"sample_{i}.wav")
        gt_path = os.path.join(gt_dir, f"gt_sample_{i}.wav")
        wav_paths.append(wav_path)
        gt_paths.append(gt_path)

        # Save ground-truth audio
        try:
            gt_audio = item["audio"]["array"]
            gt_sr = item["audio"]["sampling_rate"]
            torchaudio.save(gt_path, torch.tensor(gt_audio).unsqueeze(0), gt_sr)
        except Exception as e:
            logger.error(f"Error saving ground-truth audio {gt_path}: {e}")
            continue

        # Generate audio
        try:
            audio_stream = elevenlabs_tts(text)
            audio_data = b""
            for chunk in audio_stream:
                audio_data += chunk

            # Save MP3
            with open(mp3_path, "wb") as f:
                f.write(audio_data)

            # Convert MP3 to WAV
            waveform, sr = torchaudio.load(mp3_path)
            torchaudio.save(wav_path, waveform, sr)
        except Exception as e:
            logger.error(f"Error generating/saving audio for sample {i}: {e}")
            continue

        tsv_data.append(f"{text}\tsample_{i}.wav\n")

    if not tsv_data:
        logger.error("No valid samples processed. Exiting.")
        return

    # Save TSV
    tsv_path = "eval.tsv"
    try:
        with open(tsv_path, "w") as f:
            f.writelines(tsv_data)
        logger.info(f"Saved TSV to {tsv_path}")
    except Exception as e:
        logger.error(f"Error saving TSV: {e}")
        return

    # Step 4: Compute metrics
    try:
        whisper = pipeline("automatic-speech-recognition", model="openai/whisper-small", 
                          device=0 if device == "mps" or device == "cuda" else -1)
        predictor = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True)
    except Exception as e:
        logger.error(f"Error initializing MOS/Whisper tools: {e}")
        return

    # Note: CLVP and Intelligibility skipped due to missing clvp.pth
    logger.warning(
        "Skipping CLVP and Intelligibility metrics due to missing clvp.pth. "
        "Obtain from https://github.com/neonbjb/tts-scores or contact maintainer."
    )

    # Predicted MOS
    mos_scores = []
    for wav_path in wav_paths:
        try:
            waveform, sr = torchaudio.load(wav_path)
            mos = predictor(waveform, sr=sr)
            mos_scores.append(mos.item())
        except Exception as e:
            logger.error(f"Error computing MOS for {wav_path}: {e}")
            mos_scores.append(float('inf'))
    avg_mos = sum([s for s in mos_scores if s != float('inf')]) / max(len(mos_scores), 1)
    logger.info(f"Average Predicted MOS: {avg_mos}")

    # WER and Classification Metrics
    wer_scores = []
    classification_results = []
    for i, item in enumerate([item for _, item in valid_samples]):
        try:
            wav_path = wav_paths[i]
            pred_text = whisper(wav_path)["text"].strip().lower()
            gt_text = item["text"].strip().lower()
            wer_score = wer(gt_text, pred_text)
            wer_scores.append(wer_score)
            # Compute TP, FP, FN, TN
            metrics = compute_classification_metrics(gt_text, pred_text)
            classification_results.append(metrics)
        except Exception as e:
            logger.error(f"Error computing WER/classification for sample {i}: {e}")
            wer_scores.append(float('inf'))
            classification_results.append({
                "tp": 0, "fp": 0, "fn": 0, "tn": 0,
                "precision": 0, "recall": 0, "f1": 0
            })
    avg_wer = sum([s for s in wer_scores if s != float('inf')]) / max(len(wer_scores), 1)
    logger.info(f"Average WER: {avg_wer}")

    # Aggregate classification metrics
    total_tp = sum(r["tp"] for r in classification_results)
    total_fp = sum(r["fp"] for r in classification_results)
    total_fn = sum(r["fn"] for r in classification_results)
    total_tn = sum(r["tn"] for r in classification_results)
    avg_precision = sum(r["precision"] for r in classification_results) / max(len(classification_results), 1)
    avg_recall = sum(r["recall"] for r in classification_results) / max(len(classification_results), 1)
    avg_f1 = sum(r["f1"] for r in classification_results) / max(len(classification_results), 1)
    logger.info(f"Classification Metrics: TP={total_tp}, FP={total_fp}, FN={total_fn}, TN={total_tn}, "
                f"Precision={avg_precision:.3f}, Recall={avg_recall:.3f}, F1={avg_f1:.3f}")

    # MCD
    mcd_scores = []
    for gt_path, wav_path in zip(gt_paths, wav_paths):
        try:
            mcd = compute_mcd(gt_path, wav_path, sr=22050)
            mcd_scores.append(mcd)
        except Exception as e:
            logger.error(f"Error computing MCD for {gt_path}, {wav_path}: {e}")
            mcd_scores.append(float('inf'))
    avg_mcd = sum([s for s in mcd_scores if s != float('inf')]) / max(len(mcd_scores), 1)
    logger.info(f"Average MCD: {avg_mcd}")

    # Save results to CSV
    results = pd.DataFrame({
        "Metric": ["CLVP", "Intelligibility", "MOS", "WER", "MCD", 
                   "True Positives", "False Positives", "False Negatives", "True Negatives",
                   "Precision", "Recall", "F1-Score"],
        "Value": [float('inf'), float('inf'), avg_mos, avg_wer, avg_mcd,
                  total_tp, total_fp, total_fn, total_tn,
                  avg_precision, avg_recall, avg_f1]
    })
    csv_path = "evaluation_results.csv"
    try:
        results.to_csv(csv_path, index=False)
        logger.info(f"Results saved to {csv_path}")
    except Exception as e:
        logger.error(f"Error saving results: {e}")

if __name__ == "__main__":
    main()