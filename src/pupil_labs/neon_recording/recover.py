import json
import logging

from upath import UPath

# from importlib.metadata import version
from pupil_labs.recover_recording.recover import RecordingFixer

log = logging.getLogger(__name__)


def is_recovery_cache_up_to_date(cache_path: UPath) -> None:
    # TODO: here, we could potentially check if the recovery was performed
    # with the most up-to-date version of pl-recover-recording

    return cache_path.exists()


def run_recovery_tool(rec_path: UPath, recover: bool | None = None) -> None:
    recovery_cache_file = rec_path / "recovery_info.json"
    if recover is None:
        recover = not is_recovery_cache_up_to_date(recovery_cache_file)

    if not recover:
        log.debug("Skipping the recovery tool")
        return

    log.info("Inspecting the recording data for potential issues")
    recovery_finished = False
    recovery_error = ""
    try:
        fixer = RecordingFixer(rec_path, cleanup_temp_files=False)
        issues = fixer.process()
        recovery_finished = True
    except Exception as e:
        log.exception("An error occurred while running the recovery tool")
        recovery_error = str(e)
        raise

    # TODO: here, we need to know not only whether any issues were found,
    # but also which of them were fixed successfully

    report = {
        # "version": version("pupil_labs.recover_recording"),
        "finished": recovery_finished,
        "error": recovery_error,
        "issues": issues,
    }
    with open(recovery_cache_file, "w+") as f:
        json.dump(report, f, indent=4)
