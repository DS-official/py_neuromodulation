"""Test the nm_resample module."""
import numpy as np
from py_neuromodulation import nm_resample


class TestResampleRaw:
    def test_upsample(self) -> None:
        """Test case where data is upsampled."""
        sfreq_old = 4000.0
        duration = 10
        times = np.linspace(0, duration, int(duration * sfreq_old))
        oscill_freqs = np.expand_dims(np.arange(10, 51, 10), axis=-1)
        data = np.sin(2 * np.pi * times * oscill_freqs)

        sfreq_new = 1000.0
        resample = nm_resample.Resample(
            sfreq_old=sfreq_old, sfreq_new=sfreq_new
        )
        data_resampled = resample.resample_raw(data)
        assert data_resampled.shape[-1] == int(duration * sfreq_new)
        # This test only works when ratio of old and new sfreq is an integer
        # It will also only work up to a certain decimal precision.
        resampled_naive = data[..., :: int(sfreq_old / sfreq_new)]
        np.testing.assert_array_almost_equal(
            data[..., :: int(sfreq_old / sfreq_new)],
            resampled_naive,
            decimal=2,
        )

    def test_downsample(self) -> None:
        """Test case where data is downsampled."""
        sfreq_old = 1000.0
        duration = 10
        times = np.linspace(0, duration, int(duration * sfreq_old))
        oscill_freqs = np.expand_dims(np.arange(10, 51, 10), axis=-1)
        data = np.sin(2 * np.pi * times * oscill_freqs)

        sfreq_new = 4000.0
        resample = nm_resample.Resample(
            sfreq_old=sfreq_old, sfreq_new=sfreq_new
        )
        data_resampled = resample.resample_raw(data)
        assert data_resampled.shape[-1] == int(duration * sfreq_new)

    def test_no_resample(self) -> None:
        """Test case where no resampling is performed."""
        sfreq_old = 1000.0
        duration = 10
        times = np.linspace(0, duration, int(duration * sfreq_old))
        oscill_freqs = np.expand_dims(np.arange(10, 51, 10), axis=-1)
        data = np.sin(2 * np.pi * times * oscill_freqs)

        sfreq_new = 1000.0
        resample = nm_resample.Resample(
            sfreq_old=sfreq_old, sfreq_new=sfreq_new
        )
        data_resampled = resample.resample_raw(data)
        np.testing.assert_array_almost_equal(data, data_resampled)
