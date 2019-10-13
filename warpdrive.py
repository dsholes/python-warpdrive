"""
Welcome to WarpDrive!

Author: Darren Sholes
Date: Aug 1, 2019

Some Notes:
- Need to break up `main` into smaller functions. It's hard to read right
now
- Is there a faster way to do this?
    - With FFT/IFFT/CrossCorrelation/Convolution?
- Need to run more tests for "robustness"
    - Shorter audio samples
    - Youtube samples of same concert

"""

import librosa
import numpy as np
import soundfile as sf
import json
import sys
from pathlib import Path

AUDIO_EXT = ['.m4a','.wav','.mp3','.flac']
RAM_LIM = 1000. # MB, completely arbitrary, just for warning
RAM_WARNING = 'Warning: Using over {0:.1f}GB RAM'.format(RAM_LIM/1e3)
WD_RESULTS_DIR = "./_warpdrive_results"

def dtw_shift_param(sig1, sig2, sr):
    """
    Find warping parameters for time shift calculation using Dynamic
    Time Warping (DTW) algorithm from `librosa` package.
    """
    # Code taken from librosa docs
    # Changed metric to 'euclidean', much more robust
    # But Why?

    x_1 = sig1
    x_2 = sig2
    n_fft = int((sr/10.)*2.)
    hop_size = int(n_fft/2.)

    x_1_chroma = librosa.feature.chroma_stft(y=x_1, sr=sr, tuning=0,
                                             norm=2, hop_length=hop_size,
                                             n_fft=n_fft)
    x_2_chroma = librosa.feature.chroma_stft(y=x_2, sr=sr, tuning=0,
                                             norm=2, hop_length=hop_size,
                                             n_fft=n_fft)

    D, wp = librosa.sequence.dtw(X=x_1_chroma, Y=x_2_chroma,
                                 metric='euclidean')
    return (wp, hop_size)

def pseudo_hist_time_shift(wp, sr, hop_size):
    """
    Build Pseudo Histogram to select "mode" of time shift data.

    Most common time shift treated as actual time shift.

    Need proper test to determine confidence in result.
    """
    tdiff_unitless = wp[:,0] - wp[:,1]
    tdiff_unique, tdiff_count = np.unique(tdiff_unitless,
                                          return_counts=True)
    tdiff_sec = tdiff_unique * hop_size / sr

    return (tdiff_sec, tdiff_count)

def find_delay_sec(sig1, sig2, sr):
    """
    Return Time Shift between signals in seconds. Note signals must
    have same sample rate
    """
    # Use Dynamic Time Warping (DTW)
    wp, hop_size = dtw_shift_param(sig1, sig2, sr)

    # Build Pseudo Histogram of time shift "guesses"
    tdiff_sec, tdiff_count = pseudo_hist_time_shift(wp, sr, hop_size)

    # Need a better confidence metric...
    count_argmax = tdiff_count.argmax()
    nearest_argmax_idx = np.array([count_argmax - 1,
                                   count_argmax,
                                   count_argmax + 1])
    nearest_counts = tdiff_count[nearest_argmax_idx]
    nearest_tdiff = tdiff_sec[nearest_argmax_idx]
    confidence = nearest_counts.sum()/tdiff_count.sum()

    # Weighted average of peak and 2 nearest neighbors
    time_shift = (nearest_tdiff*nearest_counts).sum()/nearest_counts.sum()
    return (time_shift, confidence)

    results_dict[filename]['is_base'] = False
    results_dict[filename]['fake_confidence'] = confidence
    results_dict[filename]['tshift_from_base_sec'] = time_shift


def load_audio(path_to_file):
    file = Path(path_to_file)
    data, sr = librosa.load(str(file.resolve()), sr=None)
    return AudioFile(data, sr, path=file)

class AudioFile:
    def __init__(self, data, sr, path=Path('.')):
        self.path = path
        self.data = data
        self.sr = sr
        self._update_properties()

    def _update_properties(self):
        self.dur_sec = librosa.get_duration(self.data, self.sr)
        self._arr_size_MB = self.data.nbytes/1e6 # MB


    def resample(self, sr, inplace = False):
        new_sr = sr
        new_data = librosa.resample(
                        self.data,
                        self.sr,
                        new_sr,
                        res_type = 'kaiser_fast'
                        )
        if inplace:
            self.data = new_data
            self.sr = new_sr
            self._update_properties()
        else:
            resampled_self = AudioFile(new_data, new_sr)
            return resampled_self

def main(path_to_folder):
    print("*** Engaging Warp Drive... *** \n")
    track_folder = Path(path_to_folder)
    all_files = list(track_folder.glob('*'))

    audio_files = []
    sr_list = []
    dur_list = []
    results_dict = {}
    tot_MB = 0.
    print("Loading audio files...")
    for filepath in all_files:
        if filepath.suffix in AUDIO_EXT:
            file = load_audio(filepath)
            audio_files.append(file)
            filename = file.path.stem
            sr_list.append(file.sr)
            dur_list.append(file.dur_sec)
            tot_MB += file._arr_size_MB
            if tot_MB > RAM_LIM:
                print(RAM_WARNING)
            results_dict[filename] = {'path': str(file.path.resolve()),
                                      'sr': file.sr,
                                      'dur_sec': file.dur_sec,
                                      }
        else:
            print('{0} is not supported...'.format(filepath.suffix))
            print('Skipping {0}...'.format(filepath.name))

    print("Files to process:")
    for file in audio_files:
        print('- {0}'.format(file.path.name))
    print('')

    # Get "most common" sample rate
    sr_arr = np.array(sr_list)
    sr_uniq, sr_counts = np.unique(sr_arr,return_counts=True)
    sr_common = sr_uniq[sr_counts.argmax()]

    # Because sr_uniq is sorted lowest to highest,
    # if case like np.array([48000, 48000, 44100, 44100])
    # arises, sr_counts.argmax will always return lowest sr
    # Is that desirable?

    # Resample any signals that are not already "most common" sample rate
    for file in audio_files:
        if file.sr != sr_common:
            print('For analysis only: resampling {0} from {1} to {2} Hz...'
                    .format(file.path.stem, file.sr, sr_common))
            file.resample(sr_common, inplace=True)

    # Get a "base" track to compare all other tracks
    # For now use "longest" track
    dur_arr = np.array(dur_list)
    base_file = audio_files[dur_arr.argmax()]
    base_filename = base_file.path.stem
    results_dict[base_filename]['is_base'] = True
    results_dict[base_filename]['tshift_from_base_sec'] = 0.
    print('Using {0} as base sig for comparison...'.format(base_filename))

    other_files = [file for file in audio_files
                   if file.path.stem != base_filename]

    # Run through all other tracks and compare to "base" track
    for file in other_files:
        filename = file.path.stem
        print("Analyzing {0}...".format(filename))
        sig1 = base_file.data
        sig2 = file.data
        time_shift, confidence = find_delay_sec(sig1, sig2, sr_common)
        if confidence < 0.33: # completely arbitrary cutoff
            print("")
            print("Issue between {0} and {1}".format(base_filename, filename))
            print("Are you sure they're recordings from the same session?")
            print("FakeConfidenceMetric = {0:.4f}".format(confidence))
            print("")
            raise ValueError('Somethings wrong... manually inspect this')

        results_dict[filename]['is_base'] = False
        results_dict[filename]['fake_confidence'] = confidence
        results_dict[filename]['tshift_from_base_sec'] = time_shift

    print('Finding true base signal from results...')
    tshift_list = [warp_dict['tshift_from_base_sec']
                   for warp_dict in results_dict.values()]
    tshift_arr = np.array(tshift_list)
    true_base_idx = tshift_arr.argmin()
    true_base_name = list(results_dict.keys())[true_base_idx]
    true_offset = np.abs(tshift_arr.min())

    results_folder = (track_folder / WD_RESULTS_DIR)
    results_folder.mkdir(parents=True, exist_ok=True)

    print('Padding audio to align signals...')
    for filekey, warp_dict in results_dict.items():
        sig_path = Path(warp_dict['path'])
        sig_path_str = str(sig_path.resolve())

        new_sig_name = sig_path.stem + "_warpdrive" + ".wav"
        new_sig_path = results_folder / new_sig_name
        new_sig_path_str = str(new_sig_path.resolve())

        tshift_orig = warp_dict['tshift_from_base_sec']

        sig, sr = librosa.load(sig_path_str, sr=None)

        print('Writing new audio files...')
        if filekey != true_base_name:
            tshift_new = tshift_orig + true_offset
            warp_dict['is_base'] = False
            warp_dict['tshift_from_base_sec'] = tshift_new
            num_zeros = int(tshift_new*sr)
            new_sig = np.concatenate(
                        (np.zeros(num_zeros), sig)
                      ).astype(np.float32)
            sf.write(new_sig_path_str,data=new_sig,samplerate=sr)
        else:
            warp_dict['is_base'] = True
            warp_dict['tshift_from_base_sec'] = 0.
            sf.write(new_sig_path_str,data=sig, samplerate=sr)

    print('Writing JSON metadata...')
    # Output to JSON for "metadata" file
    results_filename = "_warpdrive_{0}.json".format(track_folder.name)
    json_out_path = results_folder / results_filename
    with open(json_out_path, 'w') as fp:
        json.dump(results_dict, fp,indent=4)

    print("Wahoo! That @#$%'s been successfully warped!'")
    print("See results here: {0}".format(results_folder.resolve()))

if __name__ == "__main__":
    folder_path = sys.argv[1]
    main(folder_path)
