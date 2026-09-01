"""saved-run cache : BasinRun × Path ⇄ version-stamped JSON artifact."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import tempfile
from importlib.metadata import PackageNotFoundError, version
from os import PathLike
from pathlib import Path
from typing import Any, NoReturn

from .basin import BasinRun, Presence, Resolution, TimeAxis, WaterSeries

_FORMAT = "taqsim.saved-run"
_FORMAT_VERSION = 3
_ARTIFACT_DIGEST_FIELD = "artifact_sha256"


class SavedRunError(ValueError):
    """Base class for a saved-run artifact that cannot be loaded."""


class IncidenceVersionMismatchError(SavedRunError):
    """The cache was produced by a different incidence engine version."""


class SavedRunFormatError(SavedRunError):
    """The cache is malformed or internally inconsistent."""


def incidence_version() -> str:
    """Return the installed incidence distribution version used for cache stamps."""
    try:
        return version("incidence")
    except PackageNotFoundError as error:  # pragma: no cover - taqsim requires incidence
        raise RuntimeError("the incidence distribution is not installed") from error


def save_run(run: BasinRun, path: str | PathLike[str]) -> None:
    """Write a completed run cache atomically at an explicitly supplied path."""
    log_bytes, log_digest = run.authoritative_log()
    flows = {reach: _series_document(run.flow(reach)) for reach in sorted(run.reaches)}
    retained = {reach: _series_document(run.retained(reach)) for reach in sorted(run.reaches)}
    payload = {
        "format": _FORMAT,
        "format_version": _FORMAT_VERSION,
        "incidence_version": incidence_version(),
        "model_digest": run.model_digest,
        "resolution": run.resolution.value,
        "authoritative_log": {
            "encoding": "base64",
            "sha256": log_digest,
            "data": base64.b64encode(log_bytes).decode("ascii"),
        },
        "time": {
            "start": run.time.start.isoformat(),
            "steps": run.time.steps,
            "timestep_seconds": int(run.time.timestep.total_seconds()),
        },
        "reaches": sorted(run.reaches),
        "flows": flows,
        "retained": retained,
    }
    document = {**payload, _ARTIFACT_DIGEST_FIELD: _artifact_digest(payload)}
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=target.parent, prefix=f".{target.name}.", delete=False
        ) as stream:
            temporary = Path(stream.name)
            json.dump(document, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load_run(path: str | PathLike[str]) -> BasinRun:
    """Load a compatible cache without decoding a model document or executing incidence."""
    source = Path(path)
    try:
        with source.open(encoding="utf-8") as stream:
            raw = json.load(stream, parse_constant=_reject_json_constant)
    except (OSError, json.JSONDecodeError) as error:
        raise SavedRunFormatError(f"cannot read saved run {source}: {error}") from error
    document = _object(raw, "saved run")

    stamped_version = _string(_required(document, "incidence_version", "saved run"), "incidence_version")
    installed_version = incidence_version()
    if stamped_version != installed_version:
        raise IncidenceVersionMismatchError(
            "saved run incidence version mismatch: "
            f"artifact declares {stamped_version!r}, installed version is {installed_version!r}"
        )

    _exact_keys(
        document,
        {
            "format",
            "format_version",
            "incidence_version",
            "model_digest",
            "resolution",
            "authoritative_log",
            "time",
            "reaches",
            "flows",
            "retained",
            _ARTIFACT_DIGEST_FIELD,
        },
        "saved run",
    )
    if _string(document["format"], "format") != _FORMAT:
        _malformed(f"format must be {_FORMAT!r}")
    if _integer(document["format_version"], "format_version") != _FORMAT_VERSION:
        _malformed(f"unsupported saved-run format version {document['format_version']!r}")
    declared_digest = _string(document[_ARTIFACT_DIGEST_FIELD], _ARTIFACT_DIGEST_FIELD)
    payload = {key: value for key, value in document.items() if key != _ARTIFACT_DIGEST_FIELD}
    actual_digest = _artifact_digest(payload)
    if declared_digest != actual_digest:
        _malformed(f"artifact digest mismatch: declared {declared_digest!r}, computed {actual_digest!r}")

    model_digest = _string(document["model_digest"], "model_digest")
    try:
        resolution = Resolution(_string(document["resolution"], "resolution"))
    except ValueError as error:
        raise SavedRunFormatError(f"unknown saved-run resolution {document['resolution']!r}") from error
    log = _read_log(document["authoritative_log"])
    time = _read_time(document["time"])
    reaches = _string_list(document["reaches"], "reaches")
    if len(set(reaches)) != len(reaches):
        _malformed("reaches contains duplicate names")
    flow_documents = _object(document["flows"], "flows")
    if set(flow_documents) != set(reaches):
        _malformed("flows must contain exactly the declared reaches")
    flows = {reach: _read_series(flow_documents[reach], time, reach) for reach in reaches}
    retained_documents = _object(document["retained"], "retained")
    if set(retained_documents) != set(reaches):
        _malformed("retained must contain exactly the declared reaches")
    retained = {reach: _read_series(retained_documents[reach], time, f"retained.{reach}") for reach in reaches}
    return BasinRun._from_cache(
        model_digest=model_digest,
        authoritative_log=log,
        time=time,
        resolution=resolution,
        flows=flows,
        retained=retained,
    )


def _artifact_digest(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, allow_nan=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _series_document(series: WaterSeries) -> dict[str, list[float | str | None]]:
    return {
        "values": list(series.values),
        "presence": [state.value for state in series.presence],
    }


def _read_log(raw: Any) -> tuple[bytes, str]:
    document = _object(raw, "authoritative_log")
    _exact_keys(document, {"encoding", "sha256", "data"}, "authoritative_log")
    if _string(document["encoding"], "authoritative_log.encoding") != "base64":
        _malformed("authoritative_log.encoding must be 'base64'")
    digest = _string(document["sha256"], "authoritative_log.sha256")
    encoded = _string(document["data"], "authoritative_log.data")
    try:
        log_bytes = base64.b64decode(encoded, validate=True)
    except ValueError as error:
        raise SavedRunFormatError("authoritative_log.data is not valid base64") from error
    actual = hashlib.sha256(log_bytes).hexdigest()
    if digest != actual:
        _malformed(f"authoritative log digest mismatch: declared {digest!r}, computed {actual!r}")
    return log_bytes, digest


def _read_time(raw: Any) -> TimeAxis:
    document = _object(raw, "time")
    _exact_keys(document, {"start", "steps", "timestep_seconds"}, "time")
    start = _string(document["start"], "time.start")
    steps = _integer(document["steps"], "time.steps")
    seconds = _integer(document["timestep_seconds"], "time.timestep_seconds")
    try:
        from datetime import timedelta

        return TimeAxis(start, steps, timedelta(seconds=seconds))
    except ValueError as error:
        raise SavedRunFormatError(f"invalid saved time axis: {error}") from error


def _read_series(raw: Any, time: TimeAxis, reach: str) -> WaterSeries:
    document = _object(raw, f"flows.{reach}")
    _exact_keys(document, {"values", "presence"}, f"flows.{reach}")
    raw_values = _list(document["values"], f"flows.{reach}.values")
    raw_presence = _list(document["presence"], f"flows.{reach}.presence")
    if len(raw_values) != time.steps or len(raw_presence) != time.steps:
        _malformed(f"flow for reach {reach!r} must contain exactly {time.steps} timesteps")
    values: list[float | None] = []
    for index, value in enumerate(raw_values):
        if value is None:
            values.append(None)
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            values.append(float(value))
        else:
            _malformed(f"flows.{reach}.values[{index}] must be a number or null")
    try:
        states = tuple(Presence(_string(item, f"flows.{reach}.presence")) for item in raw_presence)
    except ValueError as error:
        raise SavedRunFormatError(f"flow for reach {reach!r} has an invalid presence state") from error
    for index, (value, state) in enumerate(zip(values, states, strict=True)):
        if state is Presence.NOT_MODELLED:
            _malformed(f"flows.{reach}[{index}] is inside the horizon but marked not_modelled")
        if state is Presence.PRESENT and value is None:
            _malformed(f"flows.{reach}[{index}] is present but has no value")
        if state is not Presence.PRESENT and value is not None:
            _malformed(f"flows.{reach}[{index}] has a value but is not present")
    dates = tuple(time.datetime_at(step) for step in range(time.steps))
    return WaterSeries(dates, tuple(values), states)


def _required(document: dict[str, Any], key: str, label: str) -> Any:
    if key not in document:
        _malformed(f"{label} is missing required field {key!r}")
    return document[key]


def _object(raw: Any, label: str) -> dict[str, Any]:
    if not isinstance(raw, dict) or not all(isinstance(key, str) for key in raw):
        _malformed(f"{label} must be an object with string keys")
    return raw


def _list(raw: Any, label: str) -> list[Any]:
    if not isinstance(raw, list):
        _malformed(f"{label} must be an array")
    return raw


def _string(raw: Any, label: str) -> str:
    if not isinstance(raw, str) or not raw:
        _malformed(f"{label} must be a non-empty string")
    return raw


def _integer(raw: Any, label: str) -> int:
    if not isinstance(raw, int) or isinstance(raw, bool):
        _malformed(f"{label} must be an integer")
    return raw


def _string_list(raw: Any, label: str) -> list[str]:
    return [_string(item, f"{label}[{index}]") for index, item in enumerate(_list(raw, label))]


def _exact_keys(document: dict[str, Any], expected: set[str], label: str) -> None:
    if set(document) != expected:
        missing = sorted(expected - set(document))
        unknown = sorted(set(document) - expected)
        _malformed(f"{label} fields differ: missing={missing}, unknown={unknown}")


def _reject_json_constant(value: str) -> NoReturn:
    raise SavedRunFormatError(f"non-finite JSON number {value!r} is not allowed")


def _malformed(message: str) -> NoReturn:
    raise SavedRunFormatError(message)
