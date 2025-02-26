from ast import literal_eval
from functools import partial
from pathlib import Path
from re import compile as re_compile
from typing import (
    Callable,
    Generic,
    Iterable,
    SupportsIndex,
    TypeVar,
    overload,
)

import numpy as np
import numpy.typing as npt

RecordT = TypeVar("RecordT", bound=np.record)


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


class Record(np.record):
    def __new__(cls, source: str | Path | np.ndarray | bytes):
        return Array.load_arrays(source, cls.dtype)[0].view(cls)

    def items(self):
        return [(k, self[k]) for k in self.dtype.fields]

    def __repr__(self):
        lines = []
        pad = "    "
        n = max(len(key) for key in self.dtype.fields)
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


class Array(np.ndarray, Generic[RecordT]):
    record_class: type = Record
    dtype: np.dtype | None = None

    def __init_subclass__(cls, record_class: type = Record):
        cls.record_class = record_class
        return super().__init_subclass__()

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
    def __getitem__(self, key: SupportsIndex) -> RecordT: ...
    @overload
    def __getitem__(self, key: slice) -> "Array": ...
    def __getitem__(self, key: SupportsIndex | slice) -> "Array | RecordT":
        result = super().__getitem__(key)
        if isinstance(result, np.void):
            if self.__class__.record_class:
                self.__class__.record_class.dtype = result.dtype
                return result.view(self.__class__.record_class)
            return result
        elif isinstance(key, slice):
            return result.view(self.__class__)
        return np.array(result)  # type: ignore

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
