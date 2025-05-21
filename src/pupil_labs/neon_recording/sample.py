from typing import Literal, TypeVar

import numpy as np
import numpy.typing as npt
import pandas as pd

from pupil_labs.video.array_like import ArrayLike

T = TypeVar("T", covariant=True)


def match_ts(
    target_ts: ArrayLike[int],
    source_ts: ArrayLike[int],
    method: Literal["nearest", "before", "after"] = "nearest",
    tolerance: int | None = None,
) -> npt.NDArray[np.int64]:
    target_ts = np.array(target_ts)
    target_df = pd.DataFrame(target_ts, columns=["target_ts"])
    target_df.index.name = "target"
    target_df = target_df.reset_index()

    source_ts = np.array(source_ts)
    source_df = pd.DataFrame(source_ts, columns=["source_ts"])
    source_df.index.name = "source"
    source_df = source_df.reset_index()

    direction_map: dict[
        Literal["nearest", "before", "after"], Literal["nearest", "backward", "forward"]
    ] = {
        "nearest": "nearest",
        "before": "backward",
        "after": "forward",
    }
    direction: Literal["nearest", "backward", "forward"] = direction_map[method]

    matching_df = pd.merge_asof(
        target_df,
        source_df,
        left_on="target_ts",
        right_on="source_ts",
        direction=direction,
        tolerance=tolerance,
    )
    source_indices = matching_df["source"].to_numpy()
    return source_indices
