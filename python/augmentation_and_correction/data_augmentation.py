import numpy as np
import numpy.random as random
import albumentations as album
import audiomentations as audio
import numpy as np
from audiomentations.core.transforms_interface import BaseSpectrogramTransform

"""
(Some) Methods and functions below are taken from Sean
"""

class UniversalCompose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data):
        if isinstance(data, dict):
            if 'image' in data:
                data = data['image']
                for t in self.transforms:
                    if isinstance(t, (album.BasicTransform, album.Compose)):
                        data = t(image=data)['image']
                    elif isinstance(t, (BaseSpectrogramTransform, audio.Compose)):
                        data = t(data)
                    elif callable(t):  # Check if the transformation is a custom function
                        data = t(data)  # Assume the custom function modifies the data in-place or returns a modified version
            elif 'signal' in data:
                sr = data['sample_rate']
                data = data['signal']
                for t in self.transforms:
                    if isinstance(t, (audio.core.transforms_interface.BaseWaveformTransform, audio.Compose)):
                        data = t(samples=data, sample_rate=sr)
                    elif callable(t):  # Check if the transformation is a custom function
                        data = t(data)  # Assume the custom function modifies the data in-place or returns a modified version
        else:
            raise TypeError("Unsupported data type. Data should be a dict containing 'image' or 'audio' keys.")
        return data

    def add_transform(self, transform):
        self.transforms.append(transform)


class ScaleAugment(object):
    def __init__(self, low_range = 0.5, up_range = 1.5):
        self.up_range = up_range  # e.g. .8
        self.low_range = low_range

    #         assert self.up_range >= self.low_range
    def __call__(self, sample):
        multiplier = np.random.uniform(self.low_range, self.up_range)
        return sample * multiplier
    

# Next two are adapted from audiomentations: https://github.com/iver56/audiomentations/tree/main/audiomentations/spec_augmentations

class SpecFrequencyMask(BaseSpectrogramTransform):
    """
    Mask a set of frequencies in a spectrogram, à la Google AI SpecAugment. This type of data
    augmentation has proved to make speech recognition models more robust.

    The masked frequencies can be replaced with either the mean of the original values or a
    given constant (e.g. zero).
    """

    supports_multichannel = True

    def __init__(
        self,
        min_mask_fraction: float = 0.03,
        max_mask_fraction: float = 0.25,
        mask_dim: int = 1,
        fill_mode: str = "constant",
        fill_constant: float = 0.0,
        p: float = 0.5,
    ):
        super().__init__(p)
        self.min_mask_fraction = min_mask_fraction
        self.max_mask_fraction = max_mask_fraction
        self.mask_dim = mask_dim
        assert fill_mode in ("mean", "constant")
        self.fill_mode = fill_mode
        self.fill_constant = fill_constant

    def randomize_parameters(self, magnitude_spectrogram):
        super().randomize_parameters(magnitude_spectrogram)
        if self.parameters["should_apply"]:
            num_frequency_bins = magnitude_spectrogram.shape[self.mask_dim]
            min_frequencies_to_mask = int(
                round(self.min_mask_fraction * num_frequency_bins)
            )
            max_frequencies_to_mask = int(
                round(self.max_mask_fraction * num_frequency_bins)
            )
            num_frequencies_to_mask = random.randint(
                min_frequencies_to_mask, max_frequencies_to_mask
            )
            self.parameters["start_frequency_index"] = random.randint(
                0, num_frequency_bins - num_frequencies_to_mask
            )
            self.parameters["end_frequency_index"] = (
                self.parameters["start_frequency_index"] + num_frequencies_to_mask
            )

    def apply(self, magnitude_spectrogram):
        if self.fill_mode == "mean":
            fill_value = np.mean(
                magnitude_spectrogram[
                self.parameters["start_frequency_index"] : self.parameters[
                    "end_frequency_index"
                ]
                ]
            )
        else:
            # self.fill_mode == "constant"
            fill_value = self.fill_constant
        magnitude_spectrogram = magnitude_spectrogram.copy()
        # magnitude_spectrogram[
        # self.parameters["start_frequency_index"] : self.parameters[
        #     "end_frequency_index"
        # ]
        # ] = fill_value
        
        # Build the slicing object
        slices = [slice(None)] * magnitude_spectrogram.ndim  # All dimensions initially take all elements
        slices[self.mask_dim] = slice(self.parameters["start_frequency_index"], self.parameters["end_frequency_index"])  # Apply slicing on the target dimension

        # Apply the mask
        magnitude_spectrogram[tuple(slices)] = fill_value
        return magnitude_spectrogram


class SpecChannelShuffle(BaseSpectrogramTransform):
    """
    Shuffle the channels of a multichannel spectrogram (channels last).
    This can help combat positional bias.
    """
    supports_multichannel = True
    supports_mono = False
    def __init__(self, p: float = 0.5, channel_dim: int = 1):
        super().__init__(p)
        self.channel_dim = channel_dim

    def randomize_parameters(self, magnitude_spectrogram):
        super().randomize_parameters(magnitude_spectrogram)
        if self.parameters["should_apply"]:
            self.parameters["shuffled_channel_indexes"] = list(range(magnitude_spectrogram.shape[self.channel_dim]))
            random.shuffle(self.parameters["shuffled_channel_indexes"])

    def apply(self, magnitude_spectrogram):
        return np.take(magnitude_spectrogram, self.parameters["shuffled_channel_indexes"], axis=self.channel_dim)
    

# ! Below functions are not debugged yet
class Jitter(object):
    """
    randomly select the default window from the original window
    scale the amt of jitter by jitter amt
    validation: just return the default window.
    """

    def __init__(self, original_window, default_window, jitter_amt, sr=200, decimation=6, validate=False):
        self.original_window = original_window
        self.default_window = default_window
        self.jitter_scale = jitter_amt

        default_samples = np.asarray(default_window) - self.original_window[0]
        default_samples = np.asarray(default_samples) * sr / decimation

        default_samples[0] = int(default_samples[0])
        default_samples[1] = int(default_samples[1])

        self.default_samples = default_samples
        self.validate = validate

        self.winsize = int(default_samples[1] - default_samples[0]) + 1
        self.max_start = int(int((original_window[1] - original_window[0]) * sr / decimation) - self.winsize)

    def __call__(self, sample):
        if self.validate:
            return sample[int(self.default_samples[0]):int(self.default_samples[1]) + 1, :]
        else:
            start = np.random.randint(0, self.max_start)
            scaled_start = np.abs(start - self.default_samples[0])
            scaled_start = int(scaled_start * self.jitter_scale)
            scaled_start = int(scaled_start * np.sign(start - self.default_samples[0]) + self.default_samples[0])
            return (sample[scaled_start:scaled_start + self.winsize])


class Blackout(object):
    """
    The blackout augmentation.
    """

    def __init__(self, blackout_max_length=0.3, blackout_prob=0.5):
        self.bomax = blackout_max_length
        self.bprob = blackout_prob

    def __call__(self, sample):
        blackout_times = int(np.random.uniform(0, 1) * sample.shape[0] * self.bomax)
        start = np.random.randint(0, sample.shape[0] - sample.shape[0] * self.bomax)
        if random.uniform(0, 1) < self.bprob:
            sample[start:(start + blackout_times), :] = 0
        return sample


class AdditiveNoise(object):
    def __init__(self, sigma):
        """
        Just adds white noise.
        """
        self.sigma = sigma

    def __call__(self, sample):
        sample_ = sample + self.sigma * np.random.randn(*sample.shape)
        return sample_


class LevelChannelNoise(object):
    def __init__(self, sigma, channels=128):
        """
        Sigma: the noise std.
        """
        self.sigma = sigma
        self.channels = 128

    def __call__(self, sample):
        sample += self.sigma * np.random.randn(1, sample.shape[-1])  # Add uniform noise across the whole channel.
        return sample
