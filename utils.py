
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