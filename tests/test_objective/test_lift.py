from taqsim.objective.lift import lift
from taqsim.objective.trace import Trace


class TestLiftWithScalar:
    def test_returns_scalar_for_scalar_input(self) -> None:
        @lift
        def square(x: float) -> float:
            return x * x

        result = square(5.0)
        assert result == 25.0
        assert isinstance(result, float)

    def test_applies_function_correctly(self) -> None:
        @lift
        def add_ten(x: float) -> float:
            return x + 10

        assert add_ten(3.0) == 13.0
        assert add_ten(0.0) == 10.0
        assert add_ten(-5.0) == 5.0


class TestLiftWithTrace:
    def test_returns_trace_for_trace_input(self) -> None:
        @lift
        def double(x: float) -> float:
            return x * 2

        trace = Trace.from_dict({0: 1.0, 1: 2.0, 2: 3.0})
        result = double(trace)

        assert isinstance(result, Trace)

    def test_applies_function_to_each_value(self) -> None:
        @lift
        def triple(x: float) -> float:
            return x * 3

        trace = Trace.from_dict({0: 1.0, 1: 2.0, 2: 3.0})
        result = triple(trace)

        assert result.to_dict() == {0: 3.0, 1: 6.0, 2: 9.0}


class TestLiftMetadata:
    def test_preserves_function_name(self) -> None:
        @lift
        def my_custom_function(x: float) -> float:
            return x

        assert my_custom_function.__name__ == "my_custom_function"

    def test_preserves_docstring(self) -> None:
        @lift
        def documented_function(x: float) -> float:
            """This function has documentation."""
            return x

        assert documented_function.__doc__ == "This function has documentation."


class TestLiftComposition:
    def test_composed_lifted_functions(self) -> None:
        @lift
        def square(x: float) -> float:
            return x * x

        @lift
        def add_one(x: float) -> float:
            return x + 1

        trace = Trace.from_dict({0: 2.0, 1: 3.0})
        result = add_one(square(trace))

        assert result.to_dict() == {0: 5.0, 1: 10.0}

    def test_lifted_with_trace_arithmetic(self) -> None:
        @lift
        def halve(x: float) -> float:
            return x / 2

        trace = Trace.from_dict({0: 10.0, 1: 20.0})
        halved = halve(trace)
        result = halved + 5.0

        assert result.to_dict() == {0: 10.0, 1: 15.0}
