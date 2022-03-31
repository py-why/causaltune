from auto_causality.models.dummy import DummyModel
from auto_causality.models.wrapper import DoWhyWrapper, DoWhyMethods
from auto_causality.models.transformed_outcome import TransformedOutcomeFitter


class TransformedOutcome(DoWhyWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, inner_class=TransformedOutcomeFitter, **kwargs)


class Dummy(DoWhyWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, inner_class=DummyModel, **kwargs)
