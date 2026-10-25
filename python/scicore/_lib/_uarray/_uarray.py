"""Minimal pure-Python replacement for the deleted C ``_uarray`` extension.

Provides the uarray dispatch protocol (``BackendNotImplementedError``,
``_Function``, backend context managers) used by ``scipy._lib._uarray``.
This is a stub: the full uarray protocol can later be replaced by the
``uarray`` PyPI package or a Rust implementation.
"""

from __future__ import annotations

import contextlib
import threading


class BackendNotImplementedError(NotImplementedError):
    """Raised when no backend can handle a dispatchable call."""
    pass


class _Function:
    """A dispatchable function (uarray-compatible).

    Stores the argument extractor, replacer, domain, defaults and an
    optional default implementation.  When called, it runs the extractor to
    find dispatchable arguments, then tries registered implementations.
    """

    def __init__(self, argument_extractor, argument_replacer, domain,
                 arg_defaults=(), kw_defaults=None, default=None):
        self.argument_extractor = argument_extractor
        self.argument_replacer = argument_replacer
        self.domain = domain
        self.arg_defaults = arg_defaults or ()
        self.kw_defaults = kw_defaults or {}
        self.default = default

    def __call__(self, *args, **kwargs):
        # Dispatch to backends in priority order: context-stack backends
        # (most recent first, honoring SKIP markers), then global backends
        # (registration order, try_last entries last).  The first backend
        # that does not return NotImplemented wins; otherwise fall back to
        # the default implementation.
        for backend in self._active_backends():
            fn = getattr(backend, "__ua_function__", None)
            if fn is None:
                continue
            dom = getattr(backend, "__ua_domain__", None)
            if dom is not None and not (
                    dom == self.domain
                    or self.domain.startswith(dom + ".")
                    or dom.startswith(self.domain + ".")):
                continue
            result = fn(self, args, kwargs)
            if result is not NotImplemented:
                return result
        if self.default is not None:
            return self.default(*args, **kwargs)
        raise BackendNotImplementedError(
            f"No backend could handle {self!r}"
        )

    @staticmethod
    def _active_backends():
        """Yield candidate backends: context stack first, then globals."""
        from ._backend import _global_backends  # lazy: avoid circular import

        skipped = set()
        context = []
        for entry in reversed(_global_state.stack):
            if isinstance(entry, tuple) and entry and entry[0] == "SKIP":
                skipped.add(entry[1])
            else:
                context.append(entry)
        ordered = ([e for e in _global_backends if not e[3]]
                   + [e for e in _global_backends if e[3]])
        for entry in context + ordered:
            backend = entry[0] if isinstance(entry, tuple) else entry
            if backend not in skipped:
                yield backend

    def add_impl(self, impl):
        self.impls = getattr(self, 'impls', [])
        self.impls.append(impl)

    def __repr__(self):
        return f"_Function(domain={self.domain!r})"


class _BackendState:
    """Thread-local backend stack."""

    def __init__(self):
        self._local = threading.local()

    @property
    def stack(self):
        if not hasattr(self._local, "stack"):
            self._local.stack = []
        return self._local.stack

    def push(self, backend):
        self.stack.append(backend)

    def pop(self):
        return self.stack.pop()

    def top(self):
        if self.stack:
            return self.stack[-1]
        return None


_global_state = _BackendState()


def _pickle_state(state):
    """Serialize backend state (stack of backends)."""
    return list(state.stack)


def _unpickle_state(data, reset=False):
    """Restore backend state from serialized data."""
    _global_state.stack = list(data)


_BackendState._pickle = lambda self: _pickle_state(self)
_BackendState._unpickle = staticmethod(lambda data, reset=False: _unpickle_state(data, reset))


class _SetBackendContext:
    """Context manager that temporarily sets a backend."""

    def __init__(self, backend, coerce=False, only=False):
        self.backend = backend
        self.coerce = coerce
        self.only = only

    def __enter__(self):
        _global_state.push(self.backend)
        return self

    def __exit__(self, *exc):
        _global_state.pop()
        return False


class _SkipBackendContext:
    """Context manager that temporarily skips a backend."""

    def __init__(self, backend):
        self.backend = backend

    def __enter__(self):
        _global_state.push(("SKIP", self.backend))
        return self

    def __exit__(self, *exc):
        _global_state.pop()
        return False


__all__ = [
    "BackendNotImplementedError",
    "_Function",
    "_SkipBackendContext",
    "_SetBackendContext",
    "_BackendState",
]
