from taqsim.node.protocols import Consumes, Generates, Loses, Receives, Stores


class TestGeneratesProtocol:
    def test_class_with_generate_satisfies_protocol(self):
        class FakeSource:
            def generate(self, t: int, dt: float) -> float:
                return 100.0

        assert isinstance(FakeSource(), Generates)

    def test_class_without_generate_does_not_satisfy(self):
        class NotASource:
            pass

        assert not isinstance(NotASource(), Generates)

    def test_generate_returns_float(self):
        class ValidSource:
            def generate(self, t: int, dt: float) -> float:
                return 50.5

        source = ValidSource()
        assert source.generate(0, 1.0) == 50.5


class TestReceivesProtocol:
    def test_class_with_receive_satisfies_protocol(self):
        class FakeReceiver:
            def receive(self, amount: float, source_id: str, t: int, dt: float) -> float:
                return amount

        assert isinstance(FakeReceiver(), Receives)

    def test_class_without_receive_does_not_satisfy(self):
        class NotAReceiver:
            pass

        assert not isinstance(NotAReceiver(), Receives)

    def test_receive_returns_float(self):
        class ValidReceiver:
            def receive(self, amount: float, source_id: str, t: int, dt: float) -> float:
                return amount * 0.9

        receiver = ValidReceiver()
        assert receiver.receive(100.0, "source", 0, 1.0) == 90.0


class TestStoresProtocol:
    def test_requires_storage_property(self):
        class HasStorage:
            @property
            def storage(self) -> float:
                return 0.0

            @property
            def capacity(self) -> float:
                return 100.0

            def store(self, amount: float, t: int, dt: float) -> tuple[float, float]:
                return (amount, 0.0)

        assert isinstance(HasStorage(), Stores)

    def test_class_without_storage_does_not_satisfy(self):
        class NoStorage:
            @property
            def capacity(self) -> float:
                return 100.0

            def store(self, amount: float, t: int, dt: float) -> tuple[float, float]:
                return (amount, 0.0)

        assert not isinstance(NoStorage(), Stores)

    def test_class_without_capacity_does_not_satisfy(self):
        class NoCapacity:
            @property
            def storage(self) -> float:
                return 0.0

            def store(self, amount: float, t: int, dt: float) -> tuple[float, float]:
                return (amount, 0.0)

        assert not isinstance(NoCapacity(), Stores)

    def test_class_without_store_does_not_satisfy(self):
        class NoStore:
            @property
            def storage(self) -> float:
                return 0.0

            @property
            def capacity(self) -> float:
                return 100.0

        assert not isinstance(NoStore(), Stores)

    def test_store_returns_tuple(self):
        class ValidStore:
            @property
            def storage(self) -> float:
                return 50.0

            @property
            def capacity(self) -> float:
                return 100.0

            def store(self, amount: float, t: int, dt: float) -> tuple[float, float]:
                stored = min(amount, self.capacity - self.storage)
                overflow = max(0.0, amount - stored)
                return (stored, overflow)

        store = ValidStore()
        stored, overflow = store.store(60.0, 0, 1.0)
        assert stored == 50.0
        assert overflow == 10.0


class TestLosesProtocol:
    def test_class_with_lose_satisfies_protocol(self):
        class HasLoss:
            def lose(self, t: int, dt: float) -> float:
                return 5.0

        assert isinstance(HasLoss(), Loses)

    def test_class_without_lose_does_not_satisfy(self):
        class NoLoss:
            pass

        assert not isinstance(NoLoss(), Loses)

    def test_lose_returns_float(self):
        class ValidLoss:
            def lose(self, t: int, dt: float) -> float:
                return 2.5

        loser = ValidLoss()
        assert loser.lose(0, 1.0) == 2.5


class TestConsumesProtocol:
    def test_class_with_consume_satisfies_protocol(self):
        class HasConsumption:
            def consume(self, available: float, t: int, dt: float) -> tuple[float, float]:
                consumed = min(available, 10.0)
                remaining = available - consumed
                return (consumed, remaining)

        assert isinstance(HasConsumption(), Consumes)

    def test_class_without_consume_does_not_satisfy(self):
        class NoConsumption:
            pass

        assert not isinstance(NoConsumption(), Consumes)

    def test_consume_returns_tuple(self):
        class ValidConsumer:
            def consume(self, available: float, t: int, dt: float) -> tuple[float, float]:
                consumed = min(available, 25.0)
                remaining = available - consumed
                return (consumed, remaining)

        consumer = ValidConsumer()
        consumed, remaining = consumer.consume(100.0, 0, 1.0)
        assert consumed == 25.0
        assert remaining == 75.0


class TestProtocolCombinations:
    def test_class_can_implement_multiple_protocols(self):
        class MultiCapable:
            @property
            def storage(self) -> float:
                return 50.0

            @property
            def capacity(self) -> float:
                return 100.0

            def generate(self, t: int, dt: float) -> float:
                return 10.0

            def receive(self, amount: float, source_id: str, t: int, dt: float) -> float:
                return amount

            def store(self, amount: float, t: int, dt: float) -> tuple[float, float]:
                return (amount, 0.0)

        node = MultiCapable()
        assert isinstance(node, Generates)
        assert isinstance(node, Receives)
        assert isinstance(node, Stores)

    def test_partial_implementation_does_not_satisfy_other_protocols(self):
        class OnlyGenerates:
            def generate(self, t: int, dt: float) -> float:
                return 10.0

        node = OnlyGenerates()
        assert isinstance(node, Generates)
        assert not isinstance(node, Receives)
        assert not isinstance(node, Stores)
        assert not isinstance(node, Loses)
        assert not isinstance(node, Consumes)
