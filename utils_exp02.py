
import os
import json
import numpy as np
import librosa
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pandas as pd
from scipy import stats
from scipy.signal import resample
import soundfile as sf
from IPython.display import Audio, display
# import warnings
# warnings.filterwarnings('ignore')
import musdb
import openl3
from transformers import AutoModel, AutoFeatureExtractor
import tempfile


# ****************************************************** AUDIO MANIPULATION FUNCTIONS ******************************************************

def load_musdb_tracks(musdb_root, subset='train', max_tracks=None):
    """
    Load MUSDB18 tracks.
    Returns raw audio data without additional processing.
    """
    mus = musdb.DB(root=musdb_root, subsets=[subset])
    tracks = []
    
    for idx, track in enumerate(mus):
        if max_tracks and idx >= max_tracks:
            break
        
        # musdb returns audio in (samples, channels) format, but sometimes transposed
        # Ensure consistent (samples, channels) format
        mixture = track.audio
        vocals = track.targets['vocals'].audio
        drums = track.targets['drums'].audio
        bass = track.targets['bass'].audio
        other = track.targets['other'].audio
        
        # Transpose if needed to ensure (samples, channels) format
        if len(mixture.shape) == 2 and mixture.shape[0] < mixture.shape[1] and mixture.shape[0] <= 8:
            mixture = mixture.T
            vocals = vocals.T if len(vocals.shape) == 2 else vocals
            drums = drums.T if len(drums.shape) == 2 else drums
            bass = bass.T if len(bass.shape) == 2 else bass
            other = other.T if len(other.shape) == 2 else other
        
        track_data = {
            'name': track.name,
            'mixture': mixture,
            'vocals': vocals,
            'drums': drums,
            'bass': bass,
            'other': other
        }
        
        tracks.append(track_data)
        print(f"Loaded track {idx+1}: {track.name}")
    
    return tracks


def extract_component_audio(track, component_name):
    """Extract audio array for a specific component from track."""
    if component_name not in track:
        return None
    return track[component_name]


def extract_vocals_audio(track):
    """Extract vocals audio from track."""
    return extract_component_audio(track, 'vocals')


def extract_drums_audio(track):
    """Extract drums audio from track."""
    return extract_component_audio(track, 'drums')


def extract_bass_audio(track):
    """Extract bass audio from track."""
    return extract_component_audio(track, 'bass')


def extract_other_audio(track):
    """Extract other instruments audio from track."""
    return extract_component_audio(track, 'other')


def extract_mixture_audio(track):
    """Extract mixture (full mix) audio from track."""
    return extract_component_audio(track, 'mixture')


def extract_audio_segment_from_track(track, component_name, start_time, end_time, sample_rate=44100):
    """Extract a time segment from a specific component."""
    audio = extract_component_audio(track, component_name)
    if audio is None:
        return None
    
    # Ensure (samples, channels) format
    if len(audio.shape) == 2 and audio.shape[0] < audio.shape[1] and audio.shape[0] <= 8:
        audio = audio.T
    
    start_sample = int(start_time * sample_rate)
    end_sample = int(end_time * sample_rate)
    
    # Ensure we don't go beyond audio length
    if start_sample >= len(audio):
        return None
    if end_sample > len(audio):
        end_sample = len(audio)
    
    if len(audio.shape) > 1:
        return audio[start_sample:end_sample, :]
    else:
        return audio[start_sample:end_sample]


def convert_audio_to_mono(audio):
    """Convert audio array to mono by averaging channels."""
    if audio is None:
        return None
    if len(audio.shape) == 1:
        return audio
    elif len(audio.shape) > 1:
        # Handle both (samples, channels) and (channels, samples) formats
        if audio.shape[0] < audio.shape[1] and audio.shape[0] <= 8:
            # Likely (channels, samples) format - transpose first
            audio = audio.T
        # Now assume (samples, channels) format
        if audio.shape[1] == 1:
            return audio[:, 0]
        else:
            return np.mean(audio, axis=1)
    return audio


def extract_mono_audio(track, component_name):
    """Extract mono version of audio component."""
    audio = extract_component_audio(track, component_name)
    return convert_audio_to_mono(audio)


def normalize_audio_rms(audio, target_rms=0.1):
    """
    Normalize audio to a target RMS energy level.
    This helps eliminate the effect of different volume levels.
    
    Args:
        audio: Audio array (mono or stereo)
        target_rms: Target RMS energy level (default: 0.1)
    
    Returns:
        numpy array: Normalized audio with target RMS energy
    """
    if audio is None:
        return None
    
    # Convert to mono for RMS calculation
    mono = convert_audio_to_mono(audio)
    if mono is None or len(mono) == 0:
        return audio
    
    # Calculate current RMS
    current_rms = np.sqrt(np.mean(mono ** 2))
    
    # Avoid division by zero
    if current_rms < 1e-10:
        return audio
    
    # Calculate scaling factor
    scale_factor = target_rms / current_rms
    
    # Apply scaling to original audio (preserve stereo if applicable)
    if len(audio.shape) == 1:
        return audio * scale_factor
    else:
        # Handle stereo or multi-channel
        if len(audio.shape) == 2:
            if audio.shape[0] < audio.shape[1] and audio.shape[0] <= 8:
                # (channels, samples) format
                return audio * scale_factor
            else:
                # (samples, channels) format
                return audio * scale_factor
        return audio * scale_factor



def play_audio(audio, sample_rate=44100, max_duration_seconds=30.0):
    """
    Play audio in Jupyter notebook using IPython.display.Audio.
    Uses file-based approach to avoid IPython Audio conversion issues.
    
    Args:
        audio: Audio array (numpy array), shape (samples, channels) or (samples,)
        sample_rate: Sample rate of the audio, default: 44100
        max_duration_seconds: Maximum duration to play, longer audio will be truncated, default: 30.0
    
    Returns:
        IPython.display.Audio object
    """
    if audio is None or (isinstance(audio, np.ndarray) and audio.size == 0):
        return None
    
    audio = np.asarray(audio)
    
    # Handle shape: ensure (samples, channels) format
    if len(audio.shape) == 2:
        # If shape is (channels, samples) with channels <= 8, transpose
        if audio.shape[0] <= 8 and audio.shape[1] > audio.shape[0]:
            audio = audio.T
        # Convert to mono if stereo
        if audio.shape[1] > 1:
            audio = np.mean(audio, axis=1)
    
    # Ensure 1D array for mono audio
    if len(audio.shape) > 1:
        audio = audio.flatten()
    
    # Truncate if audio is too long
    max_samples = int(max_duration_seconds * sample_rate)
    if len(audio) > max_samples:
        audio = audio[:max_samples]
    
    # Normalize to [-1, 1] range
    max_val = np.abs(audio).max()
    if max_val > 0:
        audio = audio / max_val
    
    # Save to temporary WAV file
    tmp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    tmp_path = tmp_file.name
    tmp_file.close()
    
    sf.write(tmp_path, audio, sample_rate)
    
    # Return Audio object using filename (bypasses IPython's buggy internal conversion)
    return Audio(filename=tmp_path, rate=sample_rate)

# ****************************************************** OpenL3 FUNCTIONS ******************************************************

def extract_openl3_embedding(audio, sample_rate=44100, content_type='music', embedding_size=6144):
    """Extract OpenL3 embedding from audio array."""
    if audio is None:
        return None
    
    mono_audio = convert_audio_to_mono(audio)
    if mono_audio is None or len(mono_audio) == 0:
        return None
    
    emb, ts = openl3.get_audio_embedding(
        mono_audio,
        sample_rate,
        content_type=content_type,
        embedding_size=embedding_size
    )
    return emb


def compute_openl3_mean_embedding(embeddings):
    """Compute mean embedding from OpenL3 time-series embeddings."""
    return np.mean(embeddings, axis=0)


def extract_openl3_features_for_component(track, component_name, sample_rate=44100):
    """Extract OpenL3 features for a specific component from a track."""
    audio = extract_component_audio(track, component_name)
    if audio is None:
        return None
    embeddings = extract_openl3_embedding(audio, sample_rate=sample_rate)
    mean_embedding = compute_openl3_mean_embedding(embeddings)
    return mean_embedding


def extract_openl3_features_all_components(track, sample_rate=44100):
    """Extract OpenL3 features for all components (vocals, drums, bass, other, mixture) from a track."""
    components = ['vocals', 'drums', 'bass', 'other', 'mixture']
    features = {}
    for comp in components:
        features[comp] = extract_openl3_features_for_component(track, comp, sample_rate)
    return features


def extract_openl3_features_all_tracks(tracks, sample_rate=44100):
    """Extract OpenL3 features for all components across all tracks."""
    all_features = {
        'vocals': [],
        'drums': [],
        'bass': [],
        'other': [],
        'mixture': [],
        'track_names': []
    }
    
    for track in tracks:
        track_features = extract_openl3_features_all_components(track, sample_rate)
        for comp in ['vocals', 'drums', 'bass', 'other', 'mixture']:
            if track_features[comp] is not None:
                all_features[comp].append(track_features[comp])
            else:
                all_features[comp].append(None)
        all_features['track_names'].append(track['name'])
    
    for comp in ['vocals', 'drums', 'bass', 'other', 'mixture']:
        valid_features = [f for f in all_features[comp] if f is not None]
        if valid_features:
            try:
                # Ensure all features have compatible shapes
                all_features[comp] = np.array(valid_features)
            except (ValueError, TypeError) as e:
                # If conversion fails, set to None
                print(f"Warning: Could not convert {comp} features to array: {e}")
                all_features[comp] = None
        else:
            all_features[comp] = None
    
    return all_features


# ****************************************************** AudioMAE FUNCTIONS ******************************************************

def load_audiomae_model(model_name="MIT/ast-finetuned-audioset-10-10-0.4593"):
    """Load AudioMAE model and feature extractor."""
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    return feature_extractor, model


def resample_audio_for_audiomae(audio, current_sr, target_sr):
    """Resample audio to target sample rate for AudioMAE.
    
    Note: This function expects mono audio (1D array) since AudioMAE requires mono input.
    If stereo audio is passed, it should be converted to mono first.
    """
    if audio is None:
        return None
    
    # Audio should already be mono at this point, but handle edge cases
    if len(audio.shape) > 1:
        # If somehow stereo audio is passed, convert to mono
        if audio.shape[0] < audio.shape[1] and audio.shape[0] <= 8:
            audio = audio.T
        if audio.shape[1] > 1:
            audio = np.mean(audio, axis=1)
        else:
            audio = audio[:, 0] if len(audio.shape) == 2 else audio
    
    # Ensure 1D array
    if len(audio.shape) > 1:
        audio = audio.flatten()
    
    if current_sr != target_sr:
        return librosa.resample(audio, orig_sr=current_sr, target_sr=target_sr)
    return audio


def convert_audio_to_mono_for_audiomae(audio):
    """Convert audio to mono format for AudioMAE."""
    if audio is None:
        return None
    
    # Ensure (samples, channels) format
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


def normalize_audio_for_audiomae(audio):
    """Normalize audio to [-1, 1] range for AudioMAE."""
    if audio is None:
        return None
    max_val = np.abs(audio).max()
    if max_val > 0:
        return audio / max_val
    return audio


def extract_audiomae_embedding(audio, feature_extractor, model, sample_rate=44100):
    """Extract AudioMAE embedding from audio array."""
    # Convert to mono first (more efficient - only resample one channel)
    mono_audio = convert_audio_to_mono_for_audiomae(audio)
    
    # Then resample to target sample rate
    target_sr = feature_extractor.sampling_rate
    resampled_audio = resample_audio_for_audiomae(mono_audio, sample_rate, target_sr)
    
    # Normalize audio
    normalized_audio = normalize_audio_for_audiomae(resampled_audio)
    
    inputs = feature_extractor(normalized_audio, sampling_rate=target_sr, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    
    return embeddings


def extract_audiomae_features_for_component(track, component_name, feature_extractor, model, sample_rate=44100):
    """Extract AudioMAE features for a specific component from a track."""
    audio = extract_component_audio(track, component_name)
    if audio is None:
        return None
    embedding = extract_audiomae_embedding(audio, feature_extractor, model, sample_rate)
    return embedding


def extract_audiomae_features_all_components(track, feature_extractor, model, sample_rate=44100):
    """Extract AudioMAE features for all components (vocals, drums, bass, other, mixture) from a track."""
    components = ['vocals', 'drums', 'bass', 'other', 'mixture']
    features = {}
    for comp in components:
        features[comp] = extract_audiomae_features_for_component(track, comp, feature_extractor, model, sample_rate)
    return features


def extract_audiomae_features_all_tracks(tracks, feature_extractor, model, sample_rate=44100):
    """Extract AudioMAE features for all components across all tracks."""
    all_features = {
        'vocals': [],
        'drums': [],
        'bass': [],
        'other': [],
        'mixture': [],
        'track_names': []
    }
    
    for track in tracks:
        track_features = extract_audiomae_features_all_components(track, feature_extractor, model, sample_rate)
        for comp in ['vocals', 'drums', 'bass', 'other', 'mixture']:
            if track_features[comp] is not None:
                all_features[comp].append(track_features[comp])
            else:
                all_features[comp].append(None)
        all_features['track_names'].append(track['name'])
    
    for comp in ['vocals', 'drums', 'bass', 'other', 'mixture']:
        valid_features = [f for f in all_features[comp] if f is not None]
        if valid_features:
            try:
                # Ensure all features have compatible shapes
                all_features[comp] = np.array(valid_features)
            except (ValueError, TypeError) as e:
                # If conversion fails, set to None
                print(f"Warning: Could not convert {comp} features to array: {e}")
                all_features[comp] = None
        else:
            all_features[comp] = None
    
    return all_features


# ****************************************************** SIMILARITY COMPUTATION FUNCTIONS ******************************************************

def compute_cosine_similarity_matrix(features):
    """Compute pairwise cosine similarity matrix."""
    if features is None:
        return None
    features = np.asarray(features)
    if len(features) == 0:
        return None
    if len(features.shape) != 2:
        return None
    if features.shape[0] < 1:
        return None
    # Handle single sample case
    if features.shape[0] == 1:
        return np.array([[1.0]])
    # Check for NaN or inf
    if np.any(np.isnan(features)) or np.any(np.isinf(features)):
        # Replace NaN and inf with 0
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    return cosine_similarity(features)


def compute_euclidean_distance_matrix(features):
    """Compute pairwise Euclidean distance matrix."""
    if features is None:
        return None
    features = np.asarray(features)
    if len(features) == 0:
        return None
    if len(features.shape) != 2:
        return None
    if features.shape[0] < 1:
        return None
    # Handle single sample case
    if features.shape[0] == 1:
        return np.array([[0.0]])
    # Check for NaN or inf
    if np.any(np.isnan(features)) or np.any(np.isinf(features)):
        # Replace NaN and inf with 0
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    return euclidean_distances(features)


def compute_within_component_similarity(features_dict, component_name, metric='cosine'):
    """Compute similarity matrix within a single component."""
    if component_name not in features_dict or features_dict[component_name] is None:
        return None
    
    features = features_dict[component_name]
    if features is None:
        return None
    
    # Ensure it's a numpy array
    features = np.asarray(features)
    if len(features) == 0 or features.size == 0:
        return None
    
    if metric == 'cosine':
        return compute_cosine_similarity_matrix(features)
    elif metric == 'euclidean':
        distances = compute_euclidean_distance_matrix(features)
        if distances is None:
            return None
        return 1 / (1 + distances)
    return None


def compute_cross_component_similarity(features_dict, component1, component2, metric='cosine'):
    """Compute similarity matrix between two different components."""
    if (component1 not in features_dict or component2 not in features_dict or
        features_dict[component1] is None or features_dict[component2] is None):
        return None
    
    features1 = features_dict[component1]
    features2 = features_dict[component2]
    
    # Ensure they're numpy arrays
    features1 = np.asarray(features1)
    features2 = np.asarray(features2)
    
    if len(features1) == 0 or len(features2) == 0:
        return None
    if features1.size == 0 or features2.size == 0:
        return None
    
    # Check for NaN or inf
    if np.any(np.isnan(features1)) or np.any(np.isinf(features1)):
        features1 = np.nan_to_num(features1, nan=0.0, posinf=0.0, neginf=0.0)
    if np.any(np.isnan(features2)) or np.any(np.isinf(features2)):
        features2 = np.nan_to_num(features2, nan=0.0, posinf=0.0, neginf=0.0)
    
    if metric == 'cosine':
        return cosine_similarity(features1, features2)
    elif metric == 'euclidean':
        distances = euclidean_distances(features1, features2)
        return 1 / (1 + distances)
    return None


def compute_all_cross_component_similarities(features_dict, metric='cosine'):
    """Compute all within-component and cross-component similarity matrices."""
    components = ['vocals', 'drums', 'bass', 'other', 'mixture']
    results = {}
    
    for comp in components:
        within_sim = compute_within_component_similarity(features_dict, comp, metric)
        if within_sim is not None:
            results[f'{comp}_within'] = within_sim
    
    for i, comp1 in enumerate(components):
        for comp2 in components[i+1:]:
            cross_sim = compute_cross_component_similarity(features_dict, comp1, comp2, metric)
            if cross_sim is not None:
                results[f'{comp1}_{comp2}'] = cross_sim
    
    return results


# ****************************************************** VISUALIZATION FUNCTIONS ******************************************************

def plot_similarity_heatmap(similarity_matrix, title, component_names=None, figsize=(10, 8)):
    """Plot similarity matrix as heatmap."""
    plt.figure(figsize=figsize)
    sns.heatmap(similarity_matrix, annot=False, cmap='viridis', vmin=0, vmax=1,
                xticklabels=component_names, yticklabels=component_names)
    plt.title(title)
    plt.tight_layout()
    return plt.gcf()


def plot_cross_component_comparison(openl3_similarities, audiomae_similarities, component_pair, figsize=(15, 5)):
    """Compare cross-component similarities between OpenL3 and AudioMAE."""
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    key = f"{component_pair[0]}_{component_pair[1]}"
    
    if key in openl3_similarities:
        sns.heatmap(openl3_similarities[key], ax=axes[0], cmap='viridis', vmin=0, vmax=1, cbar=True)
        axes[0].set_title(f'OpenL3: {component_pair[0]} vs {component_pair[1]}')
    
    if key in audiomae_similarities:
        sns.heatmap(audiomae_similarities[key], ax=axes[1], cmap='viridis', vmin=0, vmax=1, cbar=True)
        axes[1].set_title(f'AudioMAE: {component_pair[0]} vs {component_pair[1]}')
    
    if key in openl3_similarities and key in audiomae_similarities:
        diff = openl3_similarities[key] - audiomae_similarities[key]
        sns.heatmap(diff, ax=axes[2], cmap='RdBu_r', center=0, cbar=True)
        axes[2].set_title('Difference (OpenL3 - AudioMAE)')
    
    plt.tight_layout()
    return fig




def plot_component_separation_comparison(openl3_similarities, audiomae_similarities, figsize=(12, 8)):
    """Compare how well models separate different components."""
    components = ['vocals', 'drums', 'bass', 'other']
    within_means_openl3 = []
    within_means_audiomae = []
    cross_means_openl3 = []
    cross_means_audiomae = []
    
    for comp in components:
        within_key = f'{comp}_within'
        if within_key in openl3_similarities:
            within_vals = openl3_similarities[within_key]
            mask = ~np.eye(within_vals.shape[0], dtype=bool)
            within_means_openl3.append(np.mean(within_vals[mask]))
        else:
            within_means_openl3.append(0)
        
        if within_key in audiomae_similarities:
            within_vals = audiomae_similarities[within_key]
            mask = ~np.eye(within_vals.shape[0], dtype=bool)
            within_means_audiomae.append(np.mean(within_vals[mask]))
        else:
            within_means_audiomae.append(0)
    
    cross_pairs = [('vocals', 'drums'), ('vocals', 'bass'), ('vocals', 'other'),
                   ('drums', 'bass'), ('drums', 'other'), ('bass', 'other')]
    
    for comp1, comp2 in cross_pairs:
        cross_key = f'{comp1}_{comp2}'
        if cross_key in openl3_similarities:
            cross_means_openl3.append(np.mean(openl3_similarities[cross_key]))
        else:
            cross_means_openl3.append(0)
        
        if cross_key in audiomae_similarities:
            cross_means_audiomae.append(np.mean(audiomae_similarities[cross_key]))
        else:
            cross_means_audiomae.append(0)
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    x_within = np.arange(len(components))
    width = 0.35
    axes[0].bar(x_within - width/2, within_means_openl3, width, label='OpenL3', alpha=0.7)
    axes[0].bar(x_within + width/2, within_means_audiomae, width, label='AudioMAE', alpha=0.7)
    axes[0].set_xlabel('Component')
    axes[0].set_ylabel('Mean Within-Component Similarity')
    axes[0].set_title('Within-Component Similarity')
    axes[0].set_xticks(x_within)
    axes[0].set_xticklabels(components)
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)
    
    x_cross = np.arange(len(cross_pairs))
    axes[1].bar(x_cross - width/2, cross_means_openl3, width, label='OpenL3', alpha=0.7)
    axes[1].bar(x_cross + width/2, cross_means_audiomae, width, label='AudioMAE', alpha=0.7)
    axes[1].set_xlabel('Component Pairs')
    axes[1].set_ylabel('Mean Cross-Component Similarity')
    axes[1].set_title('Cross-Component Similarity')
    axes[1].set_xticks(x_cross)
    axes[1].set_xticklabels([f'{c1}-{c2}' for c1, c2 in cross_pairs], rotation=45, ha='right')
    axes[1].legend()
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    return fig


# ****************************************************** EXPERIMENT 2 FUNCTIONS ******************************************************

def check_segment_has_content(audio_segment, threshold=0.01):
    """
    Check if an audio segment has meaningful content (not silence).
    
    Args:
        audio_segment: Audio array (mono or stereo)
        threshold: RMS threshold below which audio is considered silent
    
    Returns:
        bool: True if segment has content above threshold
    """
    if audio_segment is None:
        return False
    
    # Convert to mono if needed
    mono = convert_audio_to_mono(audio_segment)
    if mono is None or len(mono) == 0:
        return False
    
    # Calculate RMS energy
    rms = np.sqrt(np.mean(mono ** 2))
    return rms > threshold


def find_valid_segments(track, segment_duration=10.0, sample_rate=44100, 
                       step_size=5.0, min_rms_threshold=0.01):
    """
    Find valid 10-second segments that contain both vocals and drums.
    
    Args:
        track: Track dictionary with 'vocals', 'drums', 'mixture' keys
        segment_duration: Duration of segment in seconds (default: 10.0)
        sample_rate: Sample rate (default: 44100)
        step_size: Step size for sliding window in seconds (default: 5.0)
        min_rms_threshold: Minimum RMS threshold for content detection
    
    Returns:
        list: List of (start_time, end_time) tuples for valid segments
    """
    valid_segments = []
    
    # Get audio lengths
    mixture = extract_component_audio(track, 'mixture')
    if mixture is None:
        return valid_segments
    
    # Ensure (samples, channels) format
    if len(mixture.shape) == 2 and mixture.shape[0] < mixture.shape[1] and mixture.shape[0] <= 8:
        mixture = mixture.T
    
    total_duration = len(mixture) / sample_rate
    
    # Slide window through the track
    start_time = 0.0
    while start_time + segment_duration <= total_duration:
        end_time = start_time + segment_duration
        
        # Extract segments
        vocal_seg = extract_audio_segment_from_track(track, 'vocals', start_time, end_time, sample_rate)
        drum_seg = extract_audio_segment_from_track(track, 'drums', start_time, end_time, sample_rate)
        mixture_seg = extract_audio_segment_from_track(track, 'mixture', start_time, end_time, sample_rate)
        
        # Check if all segments have content
        if (check_segment_has_content(vocal_seg, min_rms_threshold) and
            check_segment_has_content(drum_seg, min_rms_threshold) and
            check_segment_has_content(mixture_seg, min_rms_threshold)):
            valid_segments.append((start_time, end_time))
        
        start_time += step_size
    
    return valid_segments


def extract_segment_features(segment_audio, model_type, feature_extractor=None, 
                             model=None, sample_rate=44100, normalize_rms=False, target_rms=0.1):
    """
    Extract features from an audio segment using OpenL3 or AudioMAE.
    
    Args:
        segment_audio: Audio array (mono or stereo)
        model_type: 'openl3' or 'audiomae'
        feature_extractor: AudioMAE feature extractor (required for AudioMAE)
        model: AudioMAE model (required for AudioMAE)
        sample_rate: Sample rate (default: 44100)
        normalize_rms: If True, normalize audio RMS energy before feature extraction (default: False)
        target_rms: Target RMS energy level for normalization (default: 0.1)
    
    Returns:
        numpy array: Feature embedding
    """
    if segment_audio is None:
        return None
    
    # Apply RMS normalization if requested
    if normalize_rms:
        segment_audio = normalize_audio_rms(segment_audio, target_rms=target_rms)
    
    if model_type == 'openl3':
        embedding = extract_openl3_embedding(segment_audio, sample_rate=sample_rate)
        if embedding is not None and len(embedding.shape) > 1:
            # Take mean over time dimension
            return compute_openl3_mean_embedding(embedding)
        return embedding
    
    elif model_type == 'audiomae':
        if feature_extractor is None or model is None:
            raise ValueError("feature_extractor and model required for AudioMAE")
        return extract_audiomae_embedding(segment_audio, feature_extractor, model, sample_rate)
    
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def compute_solo_vs_mixture_similarity(track, start_time, end_time, 
                                       model_type, feature_extractor=None, 
                                       model=None, sample_rate=44100, metric='cosine',
                                       normalize_rms=False, target_rms=0.1):
    """
    Compute similarity between solo component (vocals or drums) and mixture for a segment.
    
    Args:
        track: Track dictionary
        start_time: Start time of segment in seconds
        end_time: End time of segment in seconds
        model_type: 'openl3' or 'audiomae'
        feature_extractor: AudioMAE feature extractor (required for AudioMAE)
        model: AudioMAE model (required for AudioMAE)
        sample_rate: Sample rate (default: 44100)
        metric: Similarity metric ('cosine' or 'euclidean')
        normalize_rms: If True, normalize vocal and drum RMS energy before feature extraction (default: False)
        target_rms: Target RMS energy level for normalization (default: 0.1)
    
    Returns:
        dict: Dictionary with 'vocal_sim' and 'drum_sim' similarity scores
    """
    # Extract segments
    vocal_seg = extract_audio_segment_from_track(track, 'vocals', start_time, end_time, sample_rate)
    drum_seg = extract_audio_segment_from_track(track, 'drums', start_time, end_time, sample_rate)
    mixture_seg = extract_audio_segment_from_track(track, 'mixture', start_time, end_time, sample_rate)
    
    if vocal_seg is None or drum_seg is None or mixture_seg is None:
        return None
    
    # Extract features (with optional RMS normalization for vocal and drum only
    # normalize vocal and drum to the same energy level, but keep mixture as is
    vocal_feat = extract_segment_features(vocal_seg, model_type, feature_extractor, model, 
                                         sample_rate, normalize_rms=normalize_rms, target_rms=target_rms)
    drum_feat = extract_segment_features(drum_seg, model_type, feature_extractor, model, 
                                        sample_rate, normalize_rms=normalize_rms, target_rms=target_rms)
    # Mixture is not normalized to preserve its natural energy level
    mixture_feat = extract_segment_features(mixture_seg, model_type, feature_extractor, model, 
                                           sample_rate, normalize_rms=False)
    
    if vocal_feat is None or drum_feat is None or mixture_feat is None:
        return None
    
    # Reshape to 2D
    if len(vocal_feat.shape) == 1:
        vocal_feat = vocal_feat.reshape(1, -1)
    if len(drum_feat.shape) == 1:
        drum_feat = drum_feat.reshape(1, -1)
    if len(mixture_feat.shape) == 1:
        mixture_feat = mixture_feat.reshape(1, -1)
    
    # Compute similarities
    if metric == 'cosine':
        vocal_sim = cosine_similarity(vocal_feat, mixture_feat)[0, 0]
        drum_sim = cosine_similarity(drum_feat, mixture_feat)[0, 0]
    elif metric == 'euclidean':
        vocal_dist = euclidean_distances(vocal_feat, mixture_feat)[0, 0]
        drum_dist = euclidean_distances(drum_feat, mixture_feat)[0, 0]
        vocal_sim = 1 / (1 + vocal_dist)
        drum_sim = 1 / (1 + drum_dist)
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    return {
        'vocal_sim': vocal_sim,
        'drum_sim': drum_sim,
        'start_time': start_time,
        'end_time': end_time
    }


def run_experiment2(tracks, model_type, feature_extractor=None, model=None, 
                   segment_duration=10.0, sample_rate=44100, max_segments_per_track=3,
                   normalize_rms=False, target_rms=0.1):
    """
    Run Experiment 2: Solo Component vs Mixture Similarity Analysis.
    
    Args:
        tracks: List of track dictionaries
        model_type: 'openl3' or 'audiomae'
        feature_extractor: AudioMAE feature extractor (required for AudioMAE)
        model: AudioMAE model (required for AudioMAE)
        segment_duration: Duration of segments in seconds (default: 10.0)
        sample_rate: Sample rate (default: 44100)
        max_segments_per_track: Maximum number of segments to use per track (default: 3)
        normalize_rms: If True, normalize vocal and drum RMS energy before feature extraction (default: False)
        target_rms: Target RMS energy level for normalization (default: 0.1)
    
    Returns:
        dict: Results dictionary with similarities and statistics
    """
    results = {
        'track_names': [],
        'vocal_similarities': [],
        'drum_similarities': [],
        'segments_info': []
    }
    
    for track in tracks:
        track_name = track['name']
        print(f"\nProcessing track: {track_name}")
        
        # Find valid segments
        valid_segments = find_valid_segments(track, segment_duration=segment_duration, 
                                            sample_rate=sample_rate)
        
        if len(valid_segments) == 0:
            print(f"  Warning: No valid segments found for {track_name}")
            continue
        
        # Limit number of segments per track
        segments_to_use = valid_segments[:max_segments_per_track]
        print(f"  Found {len(valid_segments)} valid segments, using {len(segments_to_use)}")
        
        for start_time, end_time in segments_to_use:
            sim_result = compute_solo_vs_mixture_similarity(
                track, start_time, end_time, model_type, 
                feature_extractor, model, sample_rate,
                normalize_rms=normalize_rms, target_rms=target_rms
            )
            
            if sim_result is not None:
                results['track_names'].append(track_name)
                results['vocal_similarities'].append(sim_result['vocal_sim'])
                results['drum_similarities'].append(sim_result['drum_sim'])
                results['segments_info'].append({
                    'start': start_time,
                    'end': end_time
                })
                print(f"    Segment [{start_time:.1f}s-{end_time:.1f}s]: "
                      f"Vocal sim={sim_result['vocal_sim']:.4f}, "
                      f"Drum sim={sim_result['drum_sim']:.4f}")
    
    # Convert to numpy arrays
    results['vocal_similarities'] = np.array(results['vocal_similarities'])
    results['drum_similarities'] = np.array(results['drum_similarities'])
    
    # Compute statistics
    results['stats'] = {
        'mean_vocal_sim': np.mean(results['vocal_similarities']),
        'mean_drum_sim': np.mean(results['drum_similarities']),
        'std_vocal_sim': np.std(results['vocal_similarities']),
        'std_drum_sim': np.std(results['drum_similarities']),
        'median_vocal_sim': np.median(results['vocal_similarities']),
        'median_drum_sim': np.median(results['drum_similarities']),
        'num_segments': len(results['vocal_similarities'])
    }
    
    return results


def plot_experiment2_results(openl3_results, audiomae_results, figsize=(14, 10)):
    """
    Visualize Experiment 2 results comparing OpenL3 and AudioMAE.
    
    Args:
        openl3_results: Results dictionary from run_experiment2 for OpenL3
        audiomae_results: Results dictionary from run_experiment2 for AudioMAE
        figsize: Figure size tuple
    
    Returns:
        matplotlib figure
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    openl3_color = '#2E86AB'  # Blue
    audiomae_color = '#A23B72'  # Purple
    
    # Plot 1: Scatter plot - OpenL3
    axes[0, 0].scatter(openl3_results['vocal_similarities'], 
                       openl3_results['drum_similarities'], 
                       alpha=0.6, s=50, color=openl3_color)
    axes[0, 0].plot([0, 1], [0, 1], 'r--', alpha=0.5, label='y=x')
    axes[0, 0].set_xlabel('Vocal (solo) vs Mixture Similarity')
    axes[0, 0].set_ylabel('Drum (solo) vs Mixture Similarity')
    axes[0, 0].set_title('OpenL3: Solo Component vs Mixture Similarity')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    axes[0, 0].set_xlim([0, 1])
    axes[0, 0].set_ylim([0, 1])
    
    # Plot 2: Scatter plot - AudioMAE
    axes[0, 1].scatter(audiomae_results['vocal_similarities'], 
                       audiomae_results['drum_similarities'], 
                       alpha=0.6, s=50, color=audiomae_color)
    axes[0, 1].plot([0, 1], [0, 1], 'r--', alpha=0.5, label='y=x')
    axes[0, 1].set_xlabel('Vocal (solo) vs Mixture Similarity')
    axes[0, 1].set_ylabel('Drum (solo) vs Mixture Similarity')
    axes[0, 1].set_title('AudioMAE: Solo Component vs Mixture Similarity')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    axes[0, 1].set_xlim([0, 1])
    axes[0, 1].set_ylim([0, 1])
    
    # Plot 3: Box plot comparison
    data_to_plot = [
        openl3_results['vocal_similarities'],
        openl3_results['drum_similarities'],
        audiomae_results['vocal_similarities'],
        audiomae_results['drum_similarities']
    ]
    labels = ['OpenL3\nVocal', 'OpenL3\nDrum', 'AudioMAE\nVocal', 'AudioMAE\nDrum']
    bp = axes[1, 0].boxplot(data_to_plot, labels=labels, patch_artist=True)
    bp['boxes'][0].set_facecolor(openl3_color)
    bp['boxes'][1].set_facecolor(openl3_color)
    bp['boxes'][2].set_facecolor(audiomae_color)
    bp['boxes'][3].set_facecolor(audiomae_color)
    axes[1, 0].set_ylabel('Similarity Score')
    axes[1, 0].set_title('Similarity Distribution Comparison')
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # Plot 4: Mean comparison bar chart with error bars and value labels
    components = ['Vocal (solo)', 'Drum (solo)']
    openl3_means = [openl3_results['stats']['mean_vocal_sim'], 
                    openl3_results['stats']['mean_drum_sim']]
    audiomae_means = [audiomae_results['stats']['mean_vocal_sim'], 
                      audiomae_results['stats']['mean_drum_sim']]
    openl3_stds = [openl3_results['stats']['std_vocal_sim'], 
                   openl3_results['stats']['std_drum_sim']]
    audiomae_stds = [audiomae_results['stats']['std_vocal_sim'], 
                     audiomae_results['stats']['std_drum_sim']]
    
    x = np.arange(len(components))
    width = 0.35
    
    # Create bars with error bars
    bars1 = axes[1, 1].bar(x - width/2, openl3_means, width, 
                           yerr=openl3_stds, label='OpenL3', 
                           alpha=0.8, color=openl3_color, capsize=5, 
                           error_kw={'elinewidth': 2, 'capthick': 2})
    bars2 = axes[1, 1].bar(x + width/2, audiomae_means, width, 
                           yerr=audiomae_stds, label='AudioMAE', 
                           alpha=0.8, color=audiomae_color, capsize=5,
                           error_kw={'elinewidth': 2, 'capthick': 2})
    
    # Add value labels on top of bars
    def add_value_labels(bars, values, stds):
        for bar, val, std in zip(bars, values, stds):
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + std + 0.01,
                           f'{val:.3f}',
                           ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    add_value_labels(bars1, openl3_means, openl3_stds)
    add_value_labels(bars2, audiomae_means, audiomae_stds)
    
    # Add difference annotations
    for i, (comp, ol3_mean, am_mean) in enumerate(zip(components, openl3_means, audiomae_means)):
        diff = ol3_mean - am_mean
        max_val = max(ol3_mean + openl3_stds[i], am_mean + audiomae_stds[i])
        axes[1, 1].annotate(f'Δ={diff:+.3f}', 
                           xy=(i, max_val + 0.05), 
                           ha='center', va='bottom',
                           fontsize=8, style='italic',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))
    
    axes[1, 1].set_ylabel('Mean Similarity to Mixture', fontsize=11)
    axes[1, 1].set_title('Mean Similarity Comparison (with Error Bars)', fontsize=12, fontweight='bold')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(components, fontsize=10)
    axes[1, 1].legend(loc='upper left', fontsize=10)
    axes[1, 1].grid(axis='y', alpha=0.3, linestyle='--')
    axes[1, 1].set_ylim([0, max(max(openl3_means) + max(openl3_stds), 
                                 max(audiomae_means) + max(audiomae_stds)) * 1.2])
    
    plt.tight_layout()
    
    # Generate and print summary
    summary_text = generate_model_comparison_summary(openl3_results, audiomae_results)
    print("\n" + "=" * 80)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 80)
    print(summary_text)
    print("=" * 80 + "\n")
    
    return fig


def generate_model_comparison_summary(openl3_results, audiomae_results):
    """
    Generate a detailed text summary comparing OpenL3 and AudioMAE performance.
    
    Args:
        openl3_results: Results dictionary from run_experiment2 for OpenL3
        audiomae_results: Results dictionary from run_experiment2 for AudioMAE
    
    Returns:
        str: Formatted summary text
    """
    # Calculate statistics for OpenL3
    ol3_vocal_mean = openl3_results['stats']['mean_vocal_sim']
    ol3_vocal_std = openl3_results['stats']['std_vocal_sim']
    ol3_drum_mean = openl3_results['stats']['mean_drum_sim']
    ol3_drum_std = openl3_results['stats']['std_drum_sim']
    ol3_num_segments = openl3_results['stats']['num_segments']
    
    # Calculate statistics for AudioMAE
    am_vocal_mean = audiomae_results['stats']['mean_vocal_sim']
    am_vocal_std = audiomae_results['stats']['std_vocal_sim']
    am_drum_mean = audiomae_results['stats']['mean_drum_sim']
    am_drum_std = audiomae_results['stats']['std_drum_sim']
    am_num_segments = audiomae_results['stats']['num_segments']
    
    # Determine sensitivity for each model
    ol3_more_vocal = ol3_vocal_mean > ol3_drum_mean
    am_more_vocal = am_vocal_mean > am_drum_mean
    
    # Calculate differences
    vocal_diff = ol3_vocal_mean - am_vocal_mean
    drum_diff = ol3_drum_mean - am_drum_mean
    
    # Build summary
    summary = []
    summary.append(f"Total Segments Analyzed: OpenL3={ol3_num_segments}, AudioMAE={am_num_segments}")
    summary.append("")
    summary.append("VOCAL (Solo) vs Mixture Similarity:")
    summary.append(f"  OpenL3:  {ol3_vocal_mean:.4f} ± {ol3_vocal_std:.4f}")
    summary.append(f"  AudioMAE: {am_vocal_mean:.4f} ± {am_vocal_std:.4f}")
    summary.append(f"  Difference (OpenL3 - AudioMAE): {vocal_diff:+.4f}")
    summary.append("")
    summary.append("DRUM (Solo) vs Mixture Similarity:")
    summary.append(f"  OpenL3:  {ol3_drum_mean:.4f} ± {ol3_drum_std:.4f}")
    summary.append(f"  AudioMAE: {am_drum_mean:.4f} ± {am_drum_std:.4f}")
    summary.append(f"  Difference (OpenL3 - AudioMAE): {drum_diff:+.4f}")
    summary.append("")
    summary.append("MODEL SENSITIVITY:")
    if ol3_more_vocal:
        summary.append(f"  OpenL3:  More sensitive to VOCALS (vocal: {ol3_vocal_mean:.4f} > drum: {ol3_drum_mean:.4f})")
    else:
        summary.append(f"  OpenL3:  More sensitive to DRUMS (drum: {ol3_drum_mean:.4f} > vocal: {ol3_vocal_mean:.4f})")
    
    if am_more_vocal:
        summary.append(f"  AudioMAE: More sensitive to VOCALS (vocal: {am_vocal_mean:.4f} > drum: {am_drum_mean:.4f})")
    else:
        summary.append(f"  AudioMAE: More sensitive to DRUMS (drum: {am_drum_mean:.4f} > vocal: {am_vocal_mean:.4f})")
    summary.append("")
    summary.append("COMPARISON:")
    if abs(vocal_diff) > abs(drum_diff):
        if vocal_diff > 0:
            summary.append(f"  OpenL3 shows HIGHER vocal similarity by {abs(vocal_diff):.4f}")
        else:
            summary.append(f"  AudioMAE shows HIGHER vocal similarity by {abs(vocal_diff):.4f}")
    else:
        if drum_diff > 0:
            summary.append(f"  OpenL3 shows HIGHER drum similarity by {abs(drum_diff):.4f}")
        else:
            summary.append(f"  AudioMAE shows HIGHER drum similarity by {abs(drum_diff):.4f}")
    
    return '\n'.join(summary)


# ****************************************************** MANUAL VERIFICATION FUNCTIONS ******************************************************

def verify_segment_manually(track, start_time, end_time, sample_rate=44100):
    """
    Extract and display audio segments for manual verification.
    Returns vocal, drum, and mixture segments that can be played.
    
    Args:
        track: Track dictionary
        start_time: Start time in seconds
        end_time: End time in seconds
        sample_rate: Sample rate (default: 44100)
    
    Returns:
        dict: Dictionary with 'vocal', 'drum', 'mixture' segments and info
    """
    vocal_seg = extract_audio_segment_from_track(track, 'vocals', start_time, end_time, sample_rate)
    drum_seg = extract_audio_segment_from_track(track, 'drums', start_time, end_time, sample_rate)
    mixture_seg = extract_audio_segment_from_track(track, 'mixture', start_time, end_time, sample_rate)
    
    # Calculate RMS for each
    vocal_rms = np.sqrt(np.mean(convert_audio_to_mono(vocal_seg) ** 2)) if vocal_seg is not None else 0
    drum_rms = np.sqrt(np.mean(convert_audio_to_mono(drum_seg) ** 2)) if drum_seg is not None else 0
    mixture_rms = np.sqrt(np.mean(convert_audio_to_mono(mixture_seg) ** 2)) if mixture_seg is not None else 0
    
    return {
        'vocal': vocal_seg,
        'drum': drum_seg,
        'mixture': mixture_seg,
        'vocal_rms': vocal_rms,
        'drum_rms': drum_rms,
        'mixture_rms': mixture_rms,
        'start_time': start_time,
        'end_time': end_time,
        'duration': end_time - start_time
    }


def display_segment_for_verification(track, start_time, end_time, sample_rate=44100):
    """
    Display a segment with all three components for manual verification.
    Shows RMS values and provides playable audio.
    
    Args:
        track: Track dictionary
        start_time: Start time in seconds
        end_time: End time in seconds
        sample_rate: Sample rate (default: 44100)
    """
    seg_info = verify_segment_manually(track, start_time, end_time, sample_rate)
    
    print("=" * 60)
    print(f"Segment: {start_time:.1f}s - {end_time:.1f}s ({seg_info['duration']:.1f}s)")
    print("=" * 60)
    print(f"Vocal RMS:  {seg_info['vocal_rms']:.4f}")
    print(f"Drum RMS:   {seg_info['drum_rms']:.4f}")
    print(f"Mixture RMS: {seg_info['mixture_rms']:.4f}")
    print()
    
    print("Playing Mixture (Full Mix):")
    display(play_audio(seg_info['mixture'], sample_rate=sample_rate))
    print()
    
    print("Playing Vocal (Vocals Only):")
    display(play_audio(seg_info['vocal'], sample_rate=sample_rate))
    print()
    
    print("Playing Drum (Drums Only):")
    display(play_audio(seg_info['drum'], sample_rate=sample_rate))
    print()
    
    return seg_info


def batch_verify_segments(track, valid_segments, sample_rate=44100, max_segments=None):
    """
    Display multiple segments for batch verification.
    
    Args:
        track: Track dictionary
        valid_segments: List of (start_time, end_time) tuples
        sample_rate: Sample rate (default: 44100)
        max_segments: Maximum number of segments to display (None = all)
    
    Returns:
        list: List of segment info dictionaries
    """
    if max_segments:
        segments_to_verify = valid_segments[:max_segments]
    else:
        segments_to_verify = valid_segments
    
    all_segments_info = []
    
    for i, (start, end) in enumerate(segments_to_verify):
        print(f"\n{'='*60}")
        print(f"Segment {i+1}/{len(segments_to_verify)}")
        print(f"{'='*60}")
        seg_info = display_segment_for_verification(track, start, end, sample_rate)
        all_segments_info.append(seg_info)
        
        # Add separator
        if i < len(segments_to_verify) - 1:
            print("\n" + "="*60)
            print("Press Enter to continue to next segment...")
            print("="*60 + "\n")
    
    return all_segments_info