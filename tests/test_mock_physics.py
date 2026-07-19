"""
Phase 1 completion gate: physics correctness tests for MockBench's RC low-pass model.

Settled decisions (don't relitigate here):
- _signal_at(channel) is the source of truth: CH1 = unfiltered, phase 0.
  CH2 = _gain_at(f) applied to amplitude, _phase_at(f) = -arctan(f/fc) applied
  to phase. Everything is in Vpp.
- fc ~= 1591.5 Hz for the default MockBench() (r_ohms=10e3, c_farads=10e-9).

pytest crash course
--------------------
Discovery: pytest walks the paths you give it (here, `tests/`) for files
matching `test_*.py` or `*_test.py`, then within those files, functions
matching `test_*` (or methods on a class named `Test*`). "collected 0 items"
means one of those three naming rules didn't match -- not that your tests
failed. Nothing failed; nothing ran.

Fixtures: a function decorated with @pytest.fixture is reusable setup code.
A test "asks for" a fixture by naming it as a parameter in the test function's
signature -- pytest matches that parameter name to a fixture of the same name,
calls it, and hands the test whatever the fixture returned. You never call the
fixture function yourself. Default scope is "function" (fresh instance per
test); pass scope="module" to share one instance across a whole file.

parametrize: @pytest.mark.parametrize("name", [v1, v2, ...]) above a test
function runs that same test body once per value, each reported as its own
pass/fail -- so one bad value doesn't hide a pass on the others. Useful below
for the rolloff test (checking more than one frequency decade) or for running
the same check on both channels without copy-pasting the function.

Running: from the project root, `pytest tests/test_mock_physics.py -v`.
Because tests/__init__.py exists, pytest inserts the project root (not
tests/) onto sys.path, so `from benchagent...` imports resolve without you
setting PYTHONPATH yourself.
"""

import numpy as np
import pytest
from benchagent.drivers.mock import MockBench


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def bench():
    """A fresh MockBench with default R/C (fc ~= 1591.5 Hz), enabled."""
    bench = MockBench()
    bench.output_enable(True)
    return bench

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_gain_at_fc_is_minus_3db(bench):
    """
    At f = fc, an RC low-pass attenuates by -3 dB, i.e. gain = 1/sqrt(2) ~= 0.7071.
    This is the defining property of a cutoff frequency -- worth testing on
    its own, independent of the rolloff test below.
    """

    bench.set_sine(freq_hz=bench.fc, amplitude_vpp=4.0)
    vpp_1 = bench.measure_vpp(1)
    vpp_2 = bench.measure_vpp(2)

    gain = vpp_2 / vpp_1       # Actual Gain
    exp_value = 1 / np.sqrt(2)      # Expected Gain
    tolerance = 0.02

    assert gain == pytest.approx(exp_value, rel=tolerance)


@pytest.mark.parametrize("mult1,mult2", [(10, 100), (100, 1000), (150, 1500)])
def test_rolloff_is_20db_per_decade(bench, mult1, mult2):
    """
    Well above fc, a first-order RC low-pass rolls off at -20 dB/decade:
    gain(10*f) ~= gain(f) / 10, for f >> fc (where the "1 +" in
    1/sqrt(1+(f/fc)^2) becomes negligible next to (f/fc)^2).
    """

    f1 = mult1 * bench.fc
    bench.set_sine(freq_hz=f1, amplitude_vpp=100.0)
    gain_f1 = bench.measure_vpp(2) / bench.measure_vpp(1)  # Freq. 1 Gain

    f2 = mult2 * bench.fc
    bench.set_sine(freq_hz=f2, amplitude_vpp=100.0)
    gain_f2 = bench.measure_vpp(2) / bench.measure_vpp(1)  # Freq. 2 Gain

    gain_ratio = gain_f1 / gain_f2
    exp_value = 10
    tolerance = 0.04

    print(f" Gain 1 = {gain_f1:.4f} Gain 2 = {gain_f2: .6f} Ratio = {gain_ratio: .2f}")
    assert gain_ratio == pytest.approx(exp_value, rel=tolerance)


@pytest.mark.skip
def test_phase_at_fc_is_minus_45_degrees(bench):
    """
    At f = fc, phase = -arctan(f/fc) = -arctan(1) = -45 degrees (-pi/4 rad).

    TODO: decide what you're actually testing here -- two options:
    (a) call bench._phase_at(bench.fc) directly. It's a pure, noiseless
        function, so you can assert near-exact equality -- no noise model
        to fight with.
    (b) infer phase from CH1 vs CH2 captured waveforms (e.g. via
        cross-correlation to find the time lag, then convert to degrees).
        More realistic as an end-to-end check, meaningfully harder given
        noisy samples. Not required for Phase 1 unless you want the extra
        challenge.
    Testing the private helper directly (a) is a reasonable, defensible
    choice for a file named test_mock_physics.py -- you're still testing
    your physics, not pytest internals.
    """
    ...

@pytest.mark.skip
def test_capture_matches_measure_vpp(bench):
    """
    Self-consistency: for the same channel and settings, a captured trace's
    (max - min) should approximate measure_vpp's reading, within noise --
    two independently-computed quantities that should agree if the physics
    is applied consistently across both code paths.

    TODO:
    - set a sine, pick a channel, capture a waveform with enough timebase_s
      to cover several periods of the signal
    - compute waveform.samples.max() - waveform.samples.min()
    - compare against bench.measure_vpp(channel) for the same channel/settings
    - tolerance note: these two can disagree by MORE than noise_vrms alone
      would suggest -- see the chat discussion on why max/min over many
      samples systematically reads a bit high
    """
    ...
