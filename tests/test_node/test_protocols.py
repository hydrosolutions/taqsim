from taqsim.node.protocols import Receives
from taqsim.time import Frequency, Timestep


class TestReceivesProtocol:
    def test_class_with_receive_satisfies_protocol(self):
        class FakeReceiver:
            def receive(self, amount: float, source_id: str, t: Timestep) -> float:
                return amount

        assert isinstance(FakeReceiver(), Receives)

    def test_class_without_receive_does_not_satisfy(self):
        class NotAReceiver:
            pass

        assert not isinstance(NotAReceiver(), Receives)

    def test_receive_returns_float(self):
        class ValidReceiver:
            def receive(self, amount: float, source_id: str, t: Timestep) -> float:
                return amount * 0.9

        receiver = ValidReceiver()
        assert receiver.receive(100.0, "source", Timestep(0, Frequency.MONTHLY)) == 90.0
