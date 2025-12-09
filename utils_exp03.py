"""
Memory-efficient version of exp03 utilities.
Uses streaming processing to avoid memory overflow.
"""

import numpy as np
import librosa
import torch
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import soundfile as sf
from IPython.display import Audio
import torchopenl3
from transformers import AutoModel, AutoFeatureExtractor
import tempfile
import musdb
import gc



def load_single_track_clips(track, clip_duration=10.0, sample_rate=44100,
                            clips_per_track=1, energy_threshold_db=-40):
    """
    Load clips from a single MUSDB track (memory-efficient).

    Returns:
        List of clip dicts with drums, vocals arrays
    """
    vocals_full = track.targets['vocals'].audio
    drums_full = track.targets['drums'].audio

    # Transpose if needed
    if len(vocals_full.shape) == 2 and vocals_full.shape[0] < vocals_full.shape[1] and vocals_full.shape[0] <= 8:
        vocals_full = vocals_full.T
    if len(drums_full.shape) == 2 and drums_full.shape[0] < drums_full.shape[1] and drums_full.shape[0] <= 8:
        drums_full = drums_full.T

    # Align lengths
    min_length = min(len(vocals_full), len(drums_full))
    vocals_full = vocals_full[:min_length]
    drums_full = drums_full[:min_length]

    clip_samples = int(clip_duration * sample_rate)
    clips = []
    max_attempts = 50

    for attempt in range(max_attempts):
        if len(clips) >= clips_per_track:
            break

        # Random start position
        if min_length <= clip_samples:
            start = 0
        else:
            start = np.random.randint(0, min_length - clip_samples)

        end = start + clip_samples

        vocals_clip = vocals_full[start:end]
        drums_clip = drums_full[start:end]

        # Check energy
        vocals_ok = check_audio_energy(vocals_clip, energy_threshold_db)
        drums_ok = check_audio_energy(drums_clip, energy_threshold_db)

        if not (vocals_ok and drums_ok):
            continue

        clip_suffix = f"_clip{len(clips)+1}" if clips_per_track > 1 else ""
        clips.append({
            'name': f"{track.name}{clip_suffix}",
            'vocals': vocals_clip,
            'drums': drums_clip,
        })

    return clips


def check_audio_energy(audio, threshold_db=-40):
    """Check if audio has sufficient energy."""
    if audio is None or len(audio) == 0:
        return False

    rms = np.sqrt(np.mean(audio**2))
    if rms == 0:
        return False

    db = 20 * np.log10(rms)
    return db > threshold_db


def process_cross_mix_pairs_streaming(musdb_root, subset='train',
                                      num_pairs=100, clip_duration=10.0,
                                      clips_per_track=1,
                                      openl3_model=None, openl3_device=None,
                                      audiomae_model=None, audiomae_extractor=None, audiomae_device=None,
                                      seed=123):
    """
    Stream-process cross-mix pairs to extract features without loading all audio into memory.

    Returns:
        openl3_features, audiomae_features (list of dicts with embeddings)
    """
    np.random.seed(seed)

    # Step 1: Load track list (metadata only, very lightweight)
    mus = musdb.DB(root=musdb_root, subsets=[subset])
    all_tracks = list(mus)

    print(f"Total tracks available: {len(all_tracks)}")
    print(f"Processing {num_pairs} pairs...\n")

    openl3_features = []
    audiomae_features = []

    # Step 2: Process pairs one at a time
    for pair_idx in range(num_pairs):
        # Randomly select two different tracks
        track_indices = np.random.choice(len(all_tracks), size=2, replace=False)
        track_a_idx, track_b_idx = track_indices

        track_a_obj = all_tracks[track_a_idx]
        track_b_obj = all_tracks[track_b_idx]

        print(f"Processing pair {pair_idx+1}/{num_pairs}: {track_a_obj.name} + {track_b_obj.name}")

        # Step 3: Load clips from both tracks (only 2 tracks in memory at a time)
        clips_a = load_single_track_clips(track_a_obj, clip_duration, clips_per_track=clips_per_track)
        clips_b = load_single_track_clips(track_b_obj, clip_duration, clips_per_track=clips_per_track)

        if not clips_a or not clips_b:
            print(f"  Skipping pair {pair_idx}: No valid clips found")
            continue

        # Use first clip from each track
        track_a = clips_a[0]
        track_b = clips_b[0]

        # Step 4: Create cross-mix variants
        aa = track_a['drums'] + track_a['vocals']
        bb = track_b['drums'] + track_b['vocals']
        ab = track_a['drums'] + track_b['vocals']
        ba = track_b['drums'] + track_a['vocals']

        # Step 5: Extract features in BATCH mode (GPU-efficient!)
        pair_data = {
            'pair_idx': pair_idx,
            'track_a_name': track_a['name'],
            'track_b_name': track_b['name'],
        }

        # Prepare audio batch (4 variants)
        audio_batch = [aa, bb, ab, ba]
        variant_names = ['aa', 'bb', 'ab', 'ba']

        # OpenL3 batch extraction
        if openl3_model is not None:
            openl3_pair = pair_data.copy()
            embeddings = extract_openl3_embeddings_batch(
                audio_batch,
                sample_rate=44100,
                embedding_size=6144,
                content_type='music',
                model=openl3_model,
                batch_size=4  # Process all 4 variants together
            )
            for variant_name, embedding in zip(variant_names, embeddings):
                openl3_pair[variant_name] = embedding
            openl3_features.append(openl3_pair)

        # AudioMAE batch extraction
        if audiomae_model is not None:
            audiomae_pair = pair_data.copy()
            embeddings = extract_audiomae_embeddings_batch(
                audio_batch,
                feature_extractor=audiomae_extractor,
                model=audiomae_model,
                sample_rate=44100,
                device=audiomae_device,
                batch_size=4  # Process all 4 variants together
            )
            for variant_name, embedding in zip(variant_names, embeddings):
                audiomae_pair[variant_name] = embedding
            audiomae_features.append(audiomae_pair)

        # Step 6: Discard audio immediately to free memory
        del aa, bb, ab, ba, track_a, track_b, clips_a, clips_b

        # Force garbage collection every 10 pairs
        if (pair_idx + 1) % 10 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print(f"\nCompleted! Processed {len(openl3_features)} pairs")
    return openl3_features, audiomae_features



def convert_audio_to_mono(audio):
    """Convert audio array to mono by averaging channels."""
    if audio is None:
        return None
    if len(audio.shape) == 1:
        return audio
    elif len(audio.shape) > 1:
        if audio.shape[0] < audio.shape[1] and audio.shape[0] <= 8:
            audio = audio.T
        if audio.shape[1] == 1:
            return audio[:, 0]
        else:
            return np.mean(audio, axis=1)
    return audio


def extract_openl3_embedding(audio, sample_rate=44100, content_type='music', embedding_size=6144, model=None):
    """Extract OpenL3 embedding from audio array."""
    if audio is None:
        return None

    mono_audio = convert_audio_to_mono(audio)
    if mono_audio is None or len(mono_audio) == 0:
        return None

    emb, ts = torchopenl3.get_audio_embedding(
        mono_audio,
        sample_rate,
        model=model,
        content_type=content_type,
        embedding_size=embedding_size,
        verbose=False
    )

    if torch.is_tensor(emb):
        emb = emb.cpu().numpy()

    mean_embedding = np.mean(emb, axis=0)
    return mean_embedding


def convert_audio_to_mono_for_audiomae(audio):
    """Convert audio to mono format for AudioMAE."""
    if audio is None:
        return None

    if len(audio.shape) == 2 and audio.shape[0] < audio.shape[1] and audio.shape[0] <= 8:
        audio = audio.T

    if len(audio.shape) == 1:
        return audio
    elif len(audio.shape) == 2:
        if audio.shape[1] == 1:
            return audio[:, 0]
        else:
            return np.mean(audio, axis=1)
    return audio


def resample_audio_for_audiomae(audio, current_sr, target_sr):
    """Resample audio to target sample rate for AudioMAE."""
    if audio is None:
        return None

    if len(audio.shape) > 1:
        if audio.shape[0] < audio.shape[1] and audio.shape[0] <= 8:
            audio = audio.T
        if audio.shape[1] > 1:
            audio = np.mean(audio, axis=1)
        else:
            audio = audio[:, 0] if len(audio.shape) == 2 else audio

    if len(audio.shape) > 1:
        audio = audio.flatten()

    if current_sr != target_sr:
        return librosa.resample(audio, orig_sr=current_sr, target_sr=target_sr)
    return audio


def normalize_audio_for_audiomae(audio):
    """Normalize audio to [-1, 1] range for AudioMAE."""
    if audio is None:
        return None
    max_val = np.abs(audio).max()
    if max_val > 0:
        return audio / max_val
    return audio


def extract_audiomae_embedding(audio, feature_extractor, model, sample_rate=44100, device=None):
    """Extract AudioMAE embedding from audio array."""
    if device is None:
        device = next(model.parameters()).device

    mono_audio = convert_audio_to_mono_for_audiomae(audio)
    target_sr = feature_extractor.sampling_rate
    resampled_audio = resample_audio_for_audiomae(mono_audio, sample_rate, target_sr)
    normalized_audio = normalize_audio_for_audiomae(resampled_audio)

    inputs = feature_extractor(normalized_audio, sampling_rate=target_sr, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

    return embeddings


def analyze_cross_mix_sensitivity(cross_features, model_name='Model'):
    """
    Analyze whether the model is more sensitive to drums or vocals.
    """
    results = []

    for pair in cross_features:
        # Calculate similarities
        sim_ab_aa = cosine_similarity(
            pair['ab'].reshape(1, -1),
            pair['aa'].reshape(1, -1)
        )[0, 0]

        sim_ab_bb = cosine_similarity(
            pair['ab'].reshape(1, -1),
            pair['bb'].reshape(1, -1)
        )[0, 0]

        sim_ba_aa = cosine_similarity(
            pair['ba'].reshape(1, -1),
            pair['aa'].reshape(1, -1)
        )[0, 0]

        sim_ba_bb = cosine_similarity(
            pair['ba'].reshape(1, -1),
            pair['bb'].reshape(1, -1)
        )[0, 0]

        drums_sensitive_ab = sim_ab_aa > sim_ab_bb
        drums_sensitive_ba = sim_ba_bb > sim_ba_aa

        results.append({
            'pair_idx': pair['pair_idx'],
            'track_a': pair['track_a_name'],
            'track_b': pair['track_b_name'],
            'sim_ab_aa': sim_ab_aa,
            'sim_ab_bb': sim_ab_bb,
            'sim_ba_aa': sim_ba_aa,
            'sim_ba_bb': sim_ba_bb,
            'ab_drums_sensitive': drums_sensitive_ab,
            'ba_drums_sensitive': drums_sensitive_ba,
        })

    df = pd.DataFrame(results)

    # Statistics with effect size
    print(f"=== {model_name} Sensitivity Analysis ===\n")

    # AB analysis
    ab_drums_count = df['ab_drums_sensitive'].sum()
    ab_vocals_count = len(df) - ab_drums_count
    print(f"AB test (A drums + B vocals):")
    print(f"  More similar to AA (drums match): {ab_drums_count}/{len(df)} ({ab_drums_count/len(df)*100:.1f}%)")
    print(f"  More similar to BB (vocals match): {ab_vocals_count}/{len(df)} ({ab_vocals_count/len(df)*100:.1f}%)")

    # BA analysis
    ba_drums_count = df['ba_drums_sensitive'].sum()
    ba_vocals_count = len(df) - ba_drums_count
    print(f"\nBA test (B drums + A vocals):")
    print(f"  More similar to BB (drums match): {ba_drums_count}/{len(df)} ({ba_drums_count/len(df)*100:.1f}%)")
    print(f"  More similar to AA (vocals match): {ba_vocals_count}/{len(df)} ({ba_vocals_count/len(df)*100:.1f}%)")

    # Combined analysis
    total_drums = ab_drums_count + ba_drums_count
    total_vocals = ab_vocals_count + ba_vocals_count
    total_tests = len(df) * 2
    print(f"\nCombined results:")
    print(f"  Drums-sensitive: {total_drums}/{total_tests} ({total_drums/total_tests*100:.1f}%)")
    print(f"  Vocals-sensitive: {total_vocals}/{total_tests} ({total_vocals/total_tests*100:.1f}%)")

    # Average similarity differences (EFFECT SIZE)
    avg_ab_diff = (df['sim_ab_aa'] - df['sim_ab_bb']).mean()
    avg_ba_diff = (df['sim_ba_bb'] - df['sim_ba_aa']).mean()
    avg_combined_diff = (avg_ab_diff + avg_ba_diff) / 2

    # Standard deviation for effect size calculation
    ab_diff = df['sim_ab_aa'] - df['sim_ab_bb']
    ba_diff = df['sim_ba_bb'] - df['sim_ba_aa']
    combined_diff = pd.concat([ab_diff, ba_diff])
    std_diff = combined_diff.std()

    # Cohen's d effect size
    cohens_d = avg_combined_diff / std_diff if std_diff > 0 else 0

    print(f"\nAverage similarity differences:")
    print(f"  AB: sim(drums_match) - sim(vocals_match) = {avg_ab_diff:.4f}")
    print(f"  BA: sim(drums_match) - sim(vocals_match) = {avg_ba_diff:.4f}")
    print(f"  Combined: {avg_combined_diff:.4f}")
    print(f"  Std dev: {std_diff:.4f}")
    print(f"  Cohen's d: {cohens_d:.4f}", end="")

    # Interpret effect size
    if abs(cohens_d) < 0.2:
        effect = "(negligible)"
    elif abs(cohens_d) < 0.5:
        effect = "(small)"
    elif abs(cohens_d) < 0.8:
        effect = "(medium)"
    else:
        effect = "(large)"
    print(f" {effect}")

    if avg_combined_diff > 0.01:  # Threshold: 1% difference
        print(f"\n  → {model_name} is MORE SENSITIVE to DRUMS (meaningful difference)")
    elif avg_combined_diff < -0.01:
        print(f"\n  → {model_name} is MORE SENSITIVE to VOCALS (meaningful difference)")
    else:
        print(f"\n  → {model_name} shows NO MEANINGFUL PREFERENCE (difference < 1%)")

    return df


def load_openl3_model(content_type='music', embedding_size=6144, input_repr='mel256', use_gpu=True):
    """Load OpenL3 model with GPU support."""
    from torchopenl3.models import PytorchOpenl3

    device = torch.device('cuda' if (use_gpu and torch.cuda.is_available()) else 'cpu')
    model = PytorchOpenl3(input_repr=input_repr, content_type=content_type, embedding_size=embedding_size)
    model = model.to(device)
    model.eval()

    if device.type == 'cuda':
        print(f"OpenL3 model loaded on GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("OpenL3 model loaded on CPU")

    return model, device


def load_audiomae_model(model_name="MIT/ast-finetuned-audioset-10-10-0.4593", use_gpu=True):
    """Load AudioMAE model and feature extractor with GPU support."""
    device = torch.device('cuda' if (use_gpu and torch.cuda.is_available()) else 'cpu')

    if device.type == 'cuda':
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")

    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model = model.to(device)
    model.eval()

    if device.type == 'cuda':
        print(f"AudioMAE model loaded on GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("AudioMAE model loaded on CPU")

    return feature_extractor, model, device

# ****************************************************** BATCH PROCESSING FUNCTIONS (GPU-OPTIMIZED) ******************************************************

def extract_openl3_embeddings_batch(audio_list, sample_rate=44100, content_type='music',
                                   embedding_size=6144, model=None, batch_size=8):
    """
    Batch extract OpenL3 embeddings (GPU-efficient).

    Returns:
        List of embeddings (same order as audio_list)
    """
    if not audio_list:
        return []

    embeddings = []

    # Convert all to mono first
    batch_mono = [convert_audio_to_mono(audio) for audio in audio_list]

    # Process each audio (OpenL3 doesn't support true batching, but we optimize the loop)
    for mono_audio in batch_mono:
        if mono_audio is None or len(mono_audio) == 0:
            embeddings.append(None)
            continue

        emb, ts = torchopenl3.get_audio_embedding(
            mono_audio,
            sample_rate,
            model=model,
            content_type=content_type,
            embedding_size=embedding_size,
            verbose=False
        )

        if torch.is_tensor(emb):
            emb = emb.cpu().numpy()

        mean_embedding = np.mean(emb, axis=0)
        embeddings.append(mean_embedding)

    return embeddings


def extract_audiomae_embeddings_batch(audio_list, feature_extractor, model,
                                     sample_rate=44100, device=None, batch_size=16):
    """
    Batch extract AudioMAE embeddings (GPU-efficient with TRUE batching).

    Returns:
        List of embeddings (same order as audio_list)
    """
    if not audio_list:
        return []

    if device is None:
        device = next(model.parameters()).device

    embeddings = []

    # Process in batches
    for i in range(0, len(audio_list), batch_size):
        batch = audio_list[i:i+batch_size]

        # Prepare batch
        batch_processed = []
        for audio in batch:
            mono = convert_audio_to_mono_for_audiomae(audio)
            target_sr = feature_extractor.sampling_rate
            resampled = resample_audio_for_audiomae(mono, sample_rate, target_sr)
            normalized = normalize_audio_for_audiomae(resampled)
            batch_processed.append(normalized)

        # Batch process with feature extractor (TRUE BATCHING!)
        inputs = feature_extractor(
            batch_processed,
            sampling_rate=feature_extractor.sampling_rate,
            return_tensors="pt",
            padding=True
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            # Extract embeddings for each sample in batch
            batch_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()

        # Add to results
        for emb in batch_embeddings:
            embeddings.append(emb)

    return embeddings
