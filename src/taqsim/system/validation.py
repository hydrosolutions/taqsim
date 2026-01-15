class ValidationError(Exception):
    pass


class InsufficientLengthError(ValidationError):
    """Raised when a time-varying parameter has insufficient length for simulation."""

    def __init__(self, path: str, length: int, required: int):
        self.path = path
        self.length = length
        self.required = required
        super().__init__(
            f"Time-varying parameter '{path}' has length {length} but simulation requires {required} timesteps"
        )
