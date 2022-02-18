import json
import numpy as np
import hashlib
import pandas as pd
from flaml import AutoML


# this hashing approach was inspired by  https://death.andgravity.com/stable-hashing


def get_hash(thing):
    return hashlib.md5(json_dumps(thing).encode("utf-8")).digest().hex()


def json_dumps(thing):
    return json.dumps(
        thing,
        default=json_default,
        ensure_ascii=False,
        sort_keys=True,
        indent=None,
        separators=(",", ":"),
    )


def json_default(thing):
    if isinstance(thing, np.ndarray):
        return str(bytes(thing))
    if isinstance(thing, pd.DataFrame):
        return (tuple(thing.columns), str(bytes(thing.values)))
    if isinstance(thing, type):
        return thing.__name__
    raise TypeError(f"object of type {type(thing).__name__} not serializable")


class Memoizer(dict):
    def __call__(self, fun, x):
        key = get_hash(x)
        if key in self:
            print("Reusing cached fit!")
        else:
            print("Running new fit!")
            self[key] = fun(**x)

        return self[key]


# why bother with a borg when a closure will do? :)
memoizer = Memoizer()


def fitter_fun(fitter: type, init_args, init_kwargs, fit_args, fit_kwargs):
    x = fitter(*init_args, **init_kwargs)
    print("calling fit method with ", fit_kwargs)
    x.fit(*fit_args, **fit_kwargs)
    return x


class MemoizingWrapper(AutoML):
    def __init__(self, *args, fit_params=None, **kwargs):
        self.init_args = args
        self.init_kwargs = kwargs
        if fit_params is not None:
            self.fit_params = fit_params
        else:
            self.fit_params = {}

    def fit(self, *args, **kwargs):
        # we defer the initialization to the fit() method so we can memoize the fit
        # using all the args from both init and fit
        used_kwargs = {**kwargs, **self.fit_params}
        all_args = {
            "fitter": AutoML,
            "fit_args": args,
            "fit_kwargs": used_kwargs,
            "init_args": self.init_args,
            "init_kwargs": self.init_kwargs,
        }
        fitted = memoizer(fitter_fun, all_args)
        self.__dict__ = fitted.__dict__
