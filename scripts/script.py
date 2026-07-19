
"""Scratch script: quick sanity checks for MockBench."""

import numpy as np

from benchagent.drivers.mock import MockBench


def main():
    bench = MockBench()
    print(f"Cutoff frequency: {bench.fc:.1f} Hz")        # expect ~1591.5

    bench.set_sine(freq_hz=1000, amplitude_vpp=2.0, offset_v=0.0)

    # Vpp Check
    bench.output_enable(True)
    print(f"\nCH1 (input)  Vpp: {bench.measure_vpp(1):.3f} V")   # expect ~2.0
    print(f"CH2 (output) Vpp: {bench.measure_vpp(2):.3f} V")   # expect ~1.7

    vpp_readings = [bench.measure_vpp(2) for _ in range(5)]
    print("Five CH2 Vpp readings:", [round(r, 3) for r in vpp_readings])

    bench.output_enable(False)
    print(f"CH2 with output OFF: {bench.measure_vpp(2):.4f} V")  # expect ~0


    # Vrms Check
    bench.output_enable(True)
    print(f"\nCH1 (input)  Vrms: {bench.measure_vrms(1):.3f} V")  # expect ~0.707
    print(f"CH2 (output) Vrms: {bench.measure_vrms(2):.3f} V")  # expect ~0.599

    vrms_readings = [bench.measure_vrms(2) for _ in range(5)]
    print("Five CH2 readings:", [round(r, 3) for r in vrms_readings])

    bench.output_enable(False)
    print(f"CH2 Vrms with output OFF: {bench.measure_vrms(2):.4f} V")  # expect ~0


    # Waveform Check
    bench.output_enable(True)
    w1 = bench.capture_waveform(1, 0.01)
    print(f"\nCH1 samples: {len(w1.samples)}  rate: {w1.sample_rate_hz:.0f} Hz")  # expect 200, 20000
    print(f"CH1 peak: {np.max(np.abs(w1.samples)):.3f} V")  # expect ~1.0

    w2 = bench.capture_waveform(2, 0.01)
    print(f"\nCH2 samples: {len(w2.samples)}  rate: {w2.sample_rate_hz:.0f} Hz")  # expect 200, 20000
    print(f"CH2 peak: {np.max(np.abs(w2.samples)):.3f} V")  # expect ~0.847

    bench.output_enable(False)
    w1_off = bench.capture_waveform(1, 0.01)
    print(f"\nCH1 peak with output OFF: {np.max(np.abs(w1_off.samples)):.4f} V")  # expect ~0
    w2_off = bench.capture_waveform(2, 0.01)
    print(f"CH2 peak with output OFF: {np.max(np.abs(w2_off.samples)):.4f} V")  # expect ~0

    try:
        bench.capture_waveform(5, 0.01)
        print("ERROR: no exception raised for bad channel")
    except ValueError as e:
        print(f"Bad channel correctly rejected: {e}")


if __name__ == "__main__":
    main()