from typing import Literal, Optional, Protocol, TypeVar, Union, overload

import numpy as np
import numpy.typing as npt

from pupil_labs.matching import MatchingMethod, SampledData, SampledDataGroups
from pupil_labs.video.array_like import ArrayLike

T = TypeVar("T", covariant=True)


class NeonTimeseries(ArrayLike[T], Protocol[T]):
    @property
    def timestamps(self) -> npt.NDArray[np.int64]: ...

    @overload
    def sample(
        self,
        target_ts: ArrayLike[int],
        method: MatchingMethod = MatchingMethod.NEAREST,
        tolerance: Optional[int] = None,
        return_groups: Literal[False] = ...,
    ) -> SampledData[T]: ...
    @overload
    def sample(
        self,
        target_ts: ArrayLike[int],
        method: MatchingMethod,
        tolerance: Optional[int],
        return_groups: Literal[True],
    ) -> SampledDataGroups[T]: ...
    @overload
    def sample(
        self,
        target_ts: ArrayLike[int],
        method: MatchingMethod = MatchingMethod.NEAREST,
        tolerance: Optional[int] = None,
        *,
        return_groups: Literal[True],
    ) -> SampledDataGroups[T]: ...
    def sample(
        self,
        target_ts: ArrayLike[int],
        method: MatchingMethod = MatchingMethod.NEAREST,
        tolerance: Optional[int] = None,
        return_groups: bool = False,
    ) -> Union[SampledData[T], SampledDataGroups[T]]: ...
