import warnings

# Message emitted by dowhy >=0.13 when an EconML estimator is passed by string.
_ECONML_STRING_DEPRECATION = r".*string to specify the value for econml_estimator.*"


def _install_warning_filters() -> None:
    """Silence the one dowhy deprecation caused by causaltune's string dispatch.

    causaltune dispatches EconML estimators to dowhy by string ``method_name``
    (e.g. "backdoor.econml.dml.LinearDML"). dowhy >=0.13 deprecated passing the
    EconML estimator as a string in favour of an instance, but the string dispatch
    still works. Migrating to instance dispatch is tracked separately; silence just
    this one deprecation so it does not spam every fit. Exposed as a function so it
    can be re-applied in environments (e.g. pytest) that reset ``warnings.filters``.
    """
    warnings.filterwarnings(
        "ignore",
        message=_ECONML_STRING_DEPRECATION,
        category=DeprecationWarning,
    )


_install_warning_filters()

from causaltune.optimiser import CausalTune  # noqa: E402
from causaltune.visualizer import Visualizer  # noqa: E402
from causaltune.score.scoring import Scorer  # noqa: E402

__all__ = ["CausalTune", "Visualizer", "Scorer"]
