import numpy as np
np.set_printoptions(legacy='1.25')

from benchagent.drivers.base import FuncGen, Scope, Waveform


class MockBench(FuncGen, Scope):          # <-- MULTIPLE INHERITANCE: this class promises
    """Simulated bench: func gen -> RC low-pass -> scope."""  # to implement BOTH contracts.
                                          # Python allows inheriting from several classes;
                                          # you must implement every @abstractmethod from both,
                                          # or Python refuses to instantiate it.

    def __init__(self, r_ohms=10e3, c_farads=10e-9, noise_vrms=0.002):
        # __init__ = constructor, runs at MockBench(). The defaults after '='
        # mean callers can write MockBench() or override: MockBench(noise_vrms=0).
        # 10e3 is scientific notation: 10 * 10^3 = 10000.0 (a float).

        self.fc = 1 / (2 * np.pi * r_ohms * c_farads)   # 'self.x = ...' creates an
        self.noise_vrms = noise_vrms                    # ATTRIBUTE — data stored on this
                                                        # instance, readable in every method.
        # Internal state of the "instrument". Leading underscore = convention
        # meaning "private, don't touch from outside the class". Not enforced,
        # just a signal to readers.
        self._freq_hz = 1000.0
        self._amplitude_vpp = 1.0
        self._offset_v = 0.0
        self._output_on = False        # instruments must wake up with output OFF

    # ---- FuncGen contract -------------------------------------------------

    def set_sine(self, freq_hz, amplitude_vpp, offset_v = 0.0):
        # 'self' is always the first parameter of a method — it's the instance
        # itself. You don't pass it; bench.set_sine(1000, 2.0) fills it in.
        if freq_hz <= 0:
            raise ValueError(f"frequency must be positive, got {freq_hz}")
            # 'raise' throws an exception (like C's error return, but it
            # propagates up automatically). The f"..." is an f-string:
            # {freq_hz} gets substituted into the text.
        self._freq_hz = freq_hz
        self._amplitude_vpp = amplitude_vpp
        self._offset_v = offset_v

    def output_enable(self, on):
        self._output_on = on

    # ---- Scope contract ---------------------------------------------------


    def _gain_at(self, f):
        # Helper method (underscore = internal). THE PHYSICS GOES HERE:
        return 1 / np.sqrt(1 + (f / self.fc) ** 2)

    def _phase_at(self, f):
        return -np.arctan(f/self.fc)

    def _signal_at(self, channel):
        if channel in (1, 2):
            if channel == 1:
                return self._amplitude_vpp, 0.0
            else:
                return self._amplitude_vpp * self._gain_at(self._freq_hz), self._phase_at(self._freq_hz)
        else:
            raise ValueError(f"channel {channel} not supported")

    def _noise(self):
        # np.random.normal(mean, std_dev) -> one Gaussian random sample.
        return np.random.normal(0.0, self.noise_vrms)

    def measure_vpp(self, channel):
        if self._output_on:
            amplitude, _ = self._signal_at(channel)
            return max(0.0, amplitude + self._noise())
        else:
            return max(0.0, self._noise())

    def measure_vrms(self, channel):
        if self._output_on:
            amplitude, _ = self._signal_at(channel)
            return max(0.0, amplitude / (2 * np.sqrt(2)) + self._noise())
        else:
            return max(0.0, self._noise())

    def capture_waveform(self, channel, timebase_s):
        sample_rate_hz = 20 * self._freq_hz
        sample_size = round(sample_rate_hz * timebase_s)
        t = np.linspace(0, timebase_s, sample_size, endpoint=False)

        noise = np.random.normal(0.0, self.noise_vrms, size=t.shape)
        amplitude, phase = self._signal_at(channel)

        if self._output_on:
            signal = (amplitude / 2) * np.sin(2 * np.pi * self._freq_hz * t + phase) + self._offset_v
            waveform = signal + noise

            return Waveform(waveform, sample_rate_hz)
        else:
            signal = 0.0
            waveform = signal + noise

            return Waveform(waveform, sample_rate_hz)



