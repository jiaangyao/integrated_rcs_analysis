import numpy as np
import numpy.random as random
import albumentations as album
import audiomentations as audio
import numpy as np

"""
(Some) Methods and functions below are taken from Sean
"""

class UniversalCompose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data):
        if isinstance(data, dict):
            if 'image' in data:
                for t in self.transforms:
                    if isinstance(t, (album.BasicTransform, album.Compose)):
                        data = t(image=data['image'])['image']
                    elif callable(t):  # Check if the transformation is a custom function
                        data = t(data)  # Assume the custom function modifies the data in-place or returns a modified version
            elif 'signal' in data:
                for t in self.transforms:
                    if isinstance(t, (audio.core.transforms_interface.BaseWaveformTransform, audio.Compose)):
                        data = t(samples=data['signal'], sample_rate=data['sample_rate'])
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
