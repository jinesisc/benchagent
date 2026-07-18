
"""Scratch script: quick sanity checks for MockBench."""

from benchagent.drivers.mock import MockBench


def main():
    bench = MockBench()
    print(f"Cutoff frequency: {bench.fc:.1f} Hz")        # expect ~1591.5

    bench.set_sine(freq_hz=1000, amplitude_vpp=2.0)
    bench.output_enable(True)

    print(f"CH1 (input)  Vpp: {bench.measure_vpp(1):.3f} V")   # expect ~2.0
    print(f"CH2 (output) Vpp: {bench.measure_vpp(2):.3f} V")   # expect ~1.7

    readings = [bench.measure_vpp(2) for _ in range(5)]
    print("Five CH2 readings:", [round(r, 3) for r in readings])

    bench.output_enable(False)
    print(f"CH2 with output OFF: {bench.measure_vpp(2):.4f} V")  # expect ~0


if __name__ == "__main__":
    main()