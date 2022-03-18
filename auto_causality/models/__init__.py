from .wrapper import DoWhyWrapper
from auto_causality.transformed_outcome import TransformedOutcomeFitter


class TransformedOutcome(DoWhyWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, inner_class=TransformedOutcomeFitter, **kwargs)
