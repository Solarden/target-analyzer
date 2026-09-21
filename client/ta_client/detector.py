"""What a detector is. See implementation.md §9 and §13.

A detector is a module, not a function, because a reading is three things the runner needs:

``detect(normalized, profile) -> list[Hit]``
    The reading itself, in canonical pixels. Raises :class:`DetectorError` when it cannot
    produce one, and ``ValueError`` when it is handed something it should never be handed
    — an image that is not the profile's canonical square is a bug upstream, not a bad day.

``check(settings, profile) -> None``
    Everything ``detect`` needs that can be known before the photo is touched. The runner
    calls it before the first window opens, because a detector that refuses after forty
    clicks has thrown those clicks away.

``model_name(settings) -> str | None``
    Which model produced the reading, for the interpretation's ``model`` column. ``None``
    where the detector is the algorithm.
"""


class DetectorError(RuntimeError):
    """This detector cannot produce a reading now.

    Distinct from ``ValueError`` on purpose. A missing endpoint, a sleeping box or an
    answer that is not a reading are all conditions the run should survive — the person
    still marks the holes by hand and that reading still ships. A ``ValueError`` from a
    detector means it was called wrongly, and should stop the run.
    """
