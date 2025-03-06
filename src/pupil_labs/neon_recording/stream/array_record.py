from ast import literal_eval
from functools import partial
from pathlib import Path
from re import compile as re_compile
from typing import (
    TYPE_CHECKING,
    Callable,
    Generic,
    Iterable,
    Sequence,
    SupportsIndex,
    TypeVar,
    overload,
)

import numpy as np
import numpy.typing as npt

from pupil_labs.neon_recording.utils import unstructured

if TYPE_CHECKING:
    from pupil_labs.neon_recording.stream.stream import Stream

RecordType = TypeVar("RecordType", bound=np.record)
ArraySource = str | Path | np.ndarray | bytes


def natural_sort_key(word):
    if isinstance(word, Path):
        word = word.name
    if isinstance(word, str):
        return tuple(
            [
                int(text) if text.isdigit() else text.lower()
                for text in re_compile(r"(\d+)").split(word)
            ]
        )
    return word


class Record(np.record):
    def __new__(cls, source: str | Path | np.ndarray | bytes):
        return Array.load_arrays(source, cls.dtype)[0].view(cls)

    def items(self):
        return [(k, getattr(self, k)) for k in self.keys()]

    def keys(self):
        return self.dtype.fields

    def __repr__(self):
        lines = []
        pad = "    "
        keys = self.keys()
        n = max(len(key) for key in keys)
        for k, v in self.items():
            v_repr_lines = repr(v).splitlines()
            lines.append(f"{pad}{k:>{n}} = {v_repr_lines[0]}")
            if len(v_repr_lines) > 1:
                lines.extend(
                    f"{pad + '  '}{n * ' '}{line}" for line in v_repr_lines[1:]
                )
        lines_string = "\n".join(lines)
        lines_string = "\n" + lines_string + "\n"
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

    @overload
    def __getitem__(self, key: SupportsIndex) -> RecordType: ...
    @overload
    def __getitem__(self, key: slice | str) -> "Array": ...
    def __getitem__(self, key: SupportsIndex | slice | str) -> "Array | RecordType":
        result = super().__getitem__(key)
        if isinstance(result, np.void):
            if self.__class__.record_class:
                self.__class__.record_class.dtype = result.dtype
                return result.view(self.__class__.record_class)
            return result

        if isinstance(key, slice):
            return np.array(result).view(self.__class__)  # type: ignore
        if isinstance(key, (np.ndarray, list)):
            if isinstance(key[0], str):
                return unstructured(result)
            return np.array(result).view(self.__class__)
        return np.array(result)  # type: ignore

    def keys(self):
        return self.dtype.fields

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
            merged = decoder(item)
            parts_dtypes.add(tuple(merged.dtype.descr))
            if len(merged):
                parts.append(merged)

        if len(set(parts_dtypes)) > 1:
            raise ValueError("found multiple dtypes")
        if len(parts) == 1:
            merged = parts[0]
        elif len(parts) > 1:
            merged = np.concatenate(parts)
        else:
            merged = np.array([], dtype=np.dtype(list(parts_dtypes.pop())))
        return merged

    @classmethod
    def expand_source_arguments(cls, source: ArraySource | Iterable[ArraySource]):
        sources = []
        if isinstance(source, (str, Path, bytes, np.ndarray)):
            sources.append(source)
        elif isinstance(source, Iterable):
            sources.extend(list(source))
        else:
            raise TypeError("unknown type")
        return sorted(sources, key=natural_sort_key)


def join_struct_arrays(arrays: Sequence[npt.NDArray]):
    newdtype = [desc for a in arrays for desc in a.dtype.descr]
    newrecarray = np.empty(len(arrays[0]), dtype=newdtype)
    for a in arrays:
        assert a.dtype.names
        for name in a.dtype.names:
            newrecarray[name] = a[name]
    return newrecarray


T = TypeVar("T")


class proxy(Generic[T]):
    """
    Provides typed attribute access to key(s) on a numpy array class with IDE hints

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

    def __init__(self, columns, view: npt.DTypeLike = None):
        self.columns = [columns] if isinstance(columns, str) else columns
        self.view = view

    def __set_name__(self, owner, property_name):
        self.property_name = property_name

    @overload
    def __get__(self, obj: np.record | Record, objtype=None) -> T: ...
    @overload
    def __get__(
        self, obj: "np.ndarray | Array | Stream", objtype=None
    ) -> npt.NDArray[T]: ...
    def __get__(
        self, obj: "np.record | np.ndarray | Record | Array | Stream", objtype=None
    ) -> "T | Array[T] | npt.NDArray[T]":
        if len(self.columns) < 2:
            return obj[self.columns[0]]  # type: ignore
        elif isinstance(obj, (np.record, Record)):
            return np.array(tuple(obj[self.columns]))

        return unstructured(obj[self.columns])  # type: ignore

    def __set__(self, obj, value):
        obj[self.columns] = value
