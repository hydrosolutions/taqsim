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


class MissingAuxiliaryDataError(ValidationError):
    """Raised when a node's physical model requires auxiliary_data keys that are missing."""

    def __init__(self, node_id: str, field_name: str, model_type: str, missing_keys: frozenset[str]):
        self.node_id = node_id
        self.field_name = field_name
        self.model_type = model_type
        self.missing_keys = missing_keys
        sorted_keys = ", ".join(sorted(missing_keys))
        super().__init__(
            f"Node '{node_id}': {field_name} ({model_type}) requires auxiliary_data keys {{{sorted_keys}}} but they are missing"
        )
