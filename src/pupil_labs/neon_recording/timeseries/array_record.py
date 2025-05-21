from ast import literal_eval
from collections.abc import Callable, Iterable, Iterator, Mapping
from functools import cached_property, partial
from pathlib import Path
from re import compile as re_compile
from typing import (
    TYPE_CHECKING,
    Generic,
    SupportsIndex,
    TypeVar,
    overload,
)

import numpy as np
import numpy.typing as npt
from numpy.lib.recfunctions import structured_to_unstructured
from upath import UPath

if TYPE_CHECKING:
    from pupil_labs.neon_recording.timeseries import Timeseries

RecordType = TypeVar("RecordType", bound=np.record)
ArraySource = str | Path | np.ndarray | bytes


def natural_sort_key(word):
    if isinstance(word, Path):
        word = word.name
    if isinstance(word, str):
        return tuple([
            int(text) if text.isdigit() else text.lower()
            for text in re_compile(r"(\d+)").split(word)
        ])
    return word


def pretty_format_mapping(mapping: Mapping):
    mapping = dict(mapping)
    output = []
    pad = "    "
    keys = mapping.keys()
    n = max(len(key) for key in keys)
    for k, v in mapping.items():
        v_repr_lines = repr(v).splitlines()
        output.append(f"{pad}{k:>{n}} = {v_repr_lines[0]}")
        if len(v_repr_lines) > 1:
            output.extend(f"{pad + '  '}{n * ' '}{line}" for line in v_repr_lines[1:])
    lines_string = "\n".join(output)
    lines_string = "\n" + lines_string + "\n"
    return lines_string


class Record(np.record):
    dtype: np.dtype

    def __new__(cls, source: str | Path | np.ndarray | bytes):
        return Array.load_arrays(source, partial(Array.load_array, dtype=cls.dtype))[
            0
        ].view(cls)

    def items(self):
        return [(k, getattr(self, k, None)) for k in self.keys()]

    def keys(self):
        return list(self.dtype.names)  # type: ignore

    def __repr__(self):
        lines_string = pretty_format_mapping(self.items())
        return f"{self.__class__.__qualname__}({lines_string})"

    def __str__(self):
        lines_string = pretty_format_mapping(self.items())
        return f"{self.__class__.__qualname__}({lines_string})"


class Array(np.ndarray, Generic[RecordType]):
    record_class: type = Record
    dtype: np.dtype | None = None

    def __new__(
        cls,
        source: ArraySource | Iterable[ArraySource],
        dtype: npt.DTypeLike = None,
        fallback_dtype: npt.DTypeLike = None,
    ):
        data_dtype = dtype or cls.dtype or fallback_dtype
        sources = cls.expand_source_arguments(source)
        data = cls.load_arrays(sources, partial(cls.load_array, dtype=data_dtype))
        return data.view(cls)

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.dtype = obj.dtype
        return super().__array_finalize__(obj)

    def __iter__(self) -> Iterator[RecordType]:  # type: ignore
        for i in range(len(self)):
            yield self[i]

    def __str__(self):
        return repr(self)

    @overload  # type: ignore
    def __getitem__(self, key: SupportsIndex) -> RecordType: ...
    @overload
    def __getitem__(self, key: slice | str | list[str]) -> "Array": ...
    def __getitem__(
        self, key: SupportsIndex | slice | str | list[str]
    ) -> "Array | RecordType":
        result = super().__getitem__(key)
        if isinstance(result, np.void):
            if self.__class__.record_class:
                self.__class__.record_class.dtype = result.dtype  # type: ignore
                return result.view(self.__class__.record_class)  # type: ignore
            return result  # type: ignore

        if isinstance(key, slice):
            return np.array(result).view(self.__class__)
        if isinstance(key, (np.ndarray, list)):
            if isinstance(key[0], str):
                return unstructured(result)  # type: ignore
            return np.array(result).view(self.__class__)
        return np.array(result)  # type: ignore

    def keys(self):
        return list(self.dtype.names)  # type: ignore

    @classmethod
    def load_array(cls, source: str | Path | np.ndarray | bytes, dtype: npt.DTypeLike):
        if isinstance(source, (str, Path)):
            source_path = Path(source)
            basename = source_path.stem.split(" ps")[0]
            if source_path.suffix == ".raw":
                dtype_file = source_path.parent / f"{basename}.dtype"
                if dtype_file.exists():
                    dtype = np.dtype(literal_eval(dtype_file.read_text()))
        elif isinstance(source, np.ndarray):
            dtype = source.dtype

        if not dtype:
            source_repr = (
                source if isinstance(source, (str, Path)) else source.__class__.__name__
            )

            raise ValueError(f"dtype could not be found for {source_repr}")

        if isinstance(source, UPath):
            source = source.open("rb").read()

        if isinstance(source, (str, Path)):
            data = np.fromfile(source, dtype=dtype)
        elif isinstance(source, bytes):
            data = np.frombuffer(source, dtype=dtype)
        elif isinstance(source, np.ndarray):
            data = source
        return data

    @classmethod
    def load_arrays(
        cls,
        source: ArraySource | Iterable[ArraySource],
        decoder: Callable[[ArraySource], np.ndarray],
    ):
        parts = []
        parts_dtypes = set()
        for item in cls.expand_source_arguments(source):
            part_data = decoder(item)
            parts_dtypes.add(tuple(part_data.dtype.descr))
            if len(part_data):
                parts.append(part_data)

        if len(set(parts_dtypes)) > 1:
            raise ValueError("found multiple dtypes")
        if len(parts) == 1:
            merged = parts[0]
        elif len(parts) > 1:
            merged = np.concatenate(parts)
        elif parts_dtypes:
            merged = np.array([], dtype=np.dtype(list(parts_dtypes.pop())))
        else:
            merged = np.array([])
        return merged

    @classmethod
    def expand_source_arguments(cls, source: ArraySource | Iterable[ArraySource]):
        sources = []
        if isinstance(source, (str, Path, bytes, np.ndarray)):
            sources.append(source)
        elif isinstance(source, Iterable):
            sources.extend(list(source))  # type: ignore
        else:
            raise TypeError("unknown type")
        return sorted(sources, key=natural_sort_key)

    @cached_property
    def pd(self):
        """Return array as a pandas DataFrame"""
        import pandas

        return pandas.DataFrame(self)


T = TypeVar("T")


class fields(Generic[T]):
    """Provides typed attribute access to key(s) on a numpy array class with IDE hints

    Usage:

        >>> class GazeProps:
                ts = proxy[np.int64]('ts')
                xy = proxy[np.float32](['x', 'y'])
        >>> class GazeRecord(Record, GazeProps): ...
        >>> class GazeArray(Array, GazeProps, record_class=GazeRecord): ...
        >>> gaze = np.array(
            [
                (1741106757134448683, 100, 200),
                (1741106763470769441, 300, 400)
            ],
            dtype=[('ts', '<i8'), ('x', '<f'), ('y', '<f')]
        ).view(GazeArray)
        >>> gaze.ts
        array([100., 300.], dtype=float32)
        >>> gaze.xy
        array([[100., 200.],
               [300., 400.]], dtype=float32)
        >>> gaze[0].ts
        array(100., dtype=float32)
        >>> gaze[0].xy
        array([100., 200.], dtype=float32)
    """

    def __init__(self, columns, converter: Callable | None = None):
        self.columns = [columns] if isinstance(columns, str) else columns
        self.converter = converter

    def __set_name__(self, owner, property_name):
        self.property_name = property_name

    @overload
    def __get__(self, obj: np.record | Record, objtype=None) -> T: ...
    @overload
    def __get__(
        self, obj: "np.ndarray | Array | Timeseries", objtype=None
    ) -> npt.NDArray[T]: ...  # type: ignore
    def __get__(
        self, obj: "np.record | np.ndarray | Record | Array | Timeseries", objtype=None
    ) -> "T | Array[T] | npt.NDArray[T]":  # type: ignore
        if len(self.columns) < 2:
            result = obj[self.columns[0]]
        elif isinstance(obj, (np.record, Record)):
            try:
                result = np.array(tuple(obj[self.columns]))
            except KeyError as err:
                raise AttributeError(f"no attribute '{self.columns}'") from err
        else:
            return unstructured(obj[self.columns])  # type: ignore

        if self.converter:
            result = self.converter(result)
        return result

    def __set__(self, obj, value):
        obj[self.columns] = value


def unstructured(arr: npt.NDArray):
    if not arr.dtype.fields:
        return arr
    return structured_to_unstructured(np.array(arr))
