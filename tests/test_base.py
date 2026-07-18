from benchagent.drivers.base import FuncGen, Scope

def test_interfaces_exist():
    assert hasattr(FuncGen, "set_sine")
    assert hasattr(Scope, "measure_vpp")