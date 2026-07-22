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
from scipy.stats import norm
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
    vpp1 = bench.measure_vpp(1)
    vpp2 = bench.measure_vpp(2)

    gain = vpp2 / vpp1       # Actual Gain
    exp_value = 1 / np.sqrt(2)      # Expected Gain
    tolerance = 0.02

    assert gain == pytest.approx(exp_value, rel=tolerance)


@pytest.mark.parametrize("mult1, mult2", [(10, 100), (100, 1000), (150, 1500)])
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


def test_phase_at_fc_is_minus_45_degrees(bench):
    """
    At f = fc, phase = -arctan(f/fc) = -arctan(1) = -45 degrees (-pi/4 rad).
    """

    bench.set_sine(freq_hz=bench.fc, amplitude_vpp=4.0)

    phase_fc = bench._phase_at(bench._freq_hz)
    expected = -np.arctan(1)
    tolerance = 0.02

    assert phase_fc == pytest.approx(expected, rel=tolerance)


_CAPTURE_TEST_SEEDS = [0, 1, 2, 3, 4]
_CAPTURE_TEST_FLAKE_BUDGET = 0.001  # accept a 0.1% chance the whole suite false-fails

# Empirically calibrated via Monte Carlo (3000 unseeded trials per
# (channel, mult1) config: set_sine -> measure_vpp(channel) -> capture at
# timebase_s=0.1 -> record (max-min) - vpp; bias = sample mean of that diff,
# spread = sample std of that diff), NOT the analytic Gumbel/extreme-value
# asymptotic. That closed form (bias ~= 2*sigma*sqrt(2 ln N), even with the
# standard ln-ln correction term) still overestimated the true bias by
# ~30-40% at these N and left every diff systematically negative -- a sign
# something in the asymptotic was still wrong, not just under-toleranced.
# Simulating it directly sidesteps that.
#
# Specific to amplitude_vpp=100.0, timebase_s=0.1, and MockBench's default
# noise_vrms=0.002 -- recalibrate if any of those change.
_CAPTURE_CALIBRATION = {
    # (channel, mult1): (bias, spread)
    (1, 10):  (0.01331453, 0.00214279),
    (1, 100): (0.01363746, 0.00220903),
    (1, 150): (0.01366956, 0.00223574),
    (2, 100): (0.01579088, 0.00215400),
    (2, 150): (0.01610724, 0.00217518),
}


def _capture_tolerance(channel, mult1):
    """(bias, tolerance) for comparing (max-min) against vpp + bias, given a
    target suite-level flake budget spread across len(seeds) seeds: k solves
    p_per_seed = 1-(1-flake_budget)**(1/M), then k = two-sided z-score for
    that p."""
    bias, spread = _CAPTURE_CALIBRATION[(channel, mult1)]

    num_seeds = len(_CAPTURE_TEST_SEEDS)
    p_per_seed = 1 - (1 - _CAPTURE_TEST_FLAKE_BUDGET) ** (1 / num_seeds)
    k = norm.ppf(1 - p_per_seed / 2)

    return bias, k * spread


@pytest.mark.parametrize("mult1", [10, 100, 150])
@pytest.mark.parametrize("seed", _CAPTURE_TEST_SEEDS)
def test_capture_matches_measure_vpp_ch1(bench, mult1, seed):
    """
    Self-consistency on channel 1 (unfiltered, phase always 0): a captured
    trace's (max - min) should approximate measure_vpp's reading, within
    noise -- two independently-computed quantities that should agree if the
    physics is applied consistently across both code paths. bias/tolerance
    come from _CAPTURE_CALIBRATION (see comment above it for how).

    Seeded across several RNG draws so this isn't just one lucky noise
    realization.

    All three mult1 values are safe here (unlike the channel 2 version of
    this test): channel 1's phase is always exactly 0, and capture_waveform
    samples at fixed 18-degree steps (20 samples/period) -- 90/18=5 exactly,
    so a sample always lands dead-on the true peak regardless of frequency.
    See test_capture_matches_measure_vpp_ch2_phase for the channel where
    that alignment isn't guaranteed.
    """
    np.random.seed(seed)

    f1 = mult1 * bench.fc
    bench.set_sine(freq_hz=f1, amplitude_vpp=100.0)

    vpp1 = bench.measure_vpp(1)
    wave1 = bench.capture_waveform(1, 0.1)
    ch1_measure = wave1.samples.max() - wave1.samples.min()

    bias, tolerance = _capture_tolerance(1, mult1)
    expected = vpp1 + bias

    print(f" CH1 mult1={mult1:>3} seed={seed}  measured={ch1_measure:.6f}"
          f"  expected={expected:.6f}  diff={ch1_measure - expected:+.6f}"
          f"  tol=+/-{tolerance:.6f}")

    assert ch1_measure == pytest.approx(expected, abs=tolerance)


@pytest.mark.parametrize("mult1", [100, 150])
@pytest.mark.parametrize("seed", _CAPTURE_TEST_SEEDS)
def test_capture_matches_measure_vpp_ch2_phase(bench, mult1, seed):
    """
    Same self-consistency claim as test_capture_matches_measure_vpp_ch1, but
    on channel 2 -- this additionally exercises _phase_at, since channel 2's
    phase offset is -arctan(f/fc), not 0.

    mult1=10 is deliberately excluded: capture_waveform locks its sample
    rate to exactly 20*freq_hz (mock.py), so every period is sampled at
    fixed 18-degree steps (20 samples/period). A sample lands exactly on
    the true peak only when (90 - phase_offset) is a multiple of 18 degrees.
    At mult1=10, phase ~= -84.3 degrees, well off that alignment, giving a
    real ~4.9% undershoot at this test's timebase_s=0.1 -- a genuine
    aliasing artifact of this driver's fixed sampling, not noise, and not
    something a tolerance should be widened to hide. (Given a much longer
    capture this would eventually average out -- the rounding in
    capture_waveform's `n = round(sample_rate_hz*timebase_s)` makes the
    sample grid slowly precess -- but not within 0.1s.) mult1=100/150 give
    phases near -90 degrees, one of the aligned angles (90-(-90)=180,
    180/18=10 exactly), so the artifact vanishes there.
    """
    np.random.seed(seed)

    f1 = mult1 * bench.fc
    bench.set_sine(freq_hz=f1, amplitude_vpp=100.0)

    vpp2 = bench.measure_vpp(2)
    wave2 = bench.capture_waveform(2, 0.1)
    ch2_measure = wave2.samples.max() - wave2.samples.min()

    bias, tolerance = _capture_tolerance(2, mult1)
    expected = vpp2 + bias

    print(f" CH2 mult1={mult1:>3} seed={seed}  measured={ch2_measure:.6f}"
          f"  expected={expected:.6f}  diff={ch2_measure - expected:+.6f}"
          f"  tol=+/-{tolerance:.6f}")

    assert ch2_measure == pytest.approx(expected, abs=tolerance)



