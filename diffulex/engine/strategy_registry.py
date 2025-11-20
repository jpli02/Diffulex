from typing import Iterable

from diffulex.utils.registry import fetch_factory_name


_NOT_PROVIDED = object()


class DiffulexStrategyRegistry:
    """Registry-driven factory for module implementations."""
    
    _MODULE_MAPPING: dict[str, object] = {}
    _DEFAULT_KEY = "__default__" 
    
    @classmethod
    def register(
        cls,
        strategy_name: str,
        factory: object = _NOT_PROVIDED,
        *,
        aliases: Iterable[str] = (),
        is_default: bool = False,
        exist_ok: bool = False,
    ):
        if not isinstance(strategy_name, str) or not strategy_name:
            raise ValueError("strategy_name must be a non-empty string.")
        if isinstance(aliases, str):
            raise TypeError("aliases must be an iterable of strings, not a single string.")

        def decorator(factory_fn: object):
            cls._register(strategy_name, factory_fn, exist_ok=exist_ok)
            for alias in dict.fromkeys(aliases):
                if not isinstance(alias, str) or not alias:
                    raise ValueError("aliases must contain non-empty strings.")
                cls._register(alias, factory_fn, exist_ok=exist_ok)
            if is_default:
                cls._register(cls._DEFAULT_KEY, factory_fn, exist_ok=True)
            return factory_fn

        if factory is _NOT_PROVIDED:
            return decorator
        return decorator(factory)

    @classmethod
    def _register(cls, key: str, factory: object, *, exist_ok: bool) -> None:
        if not exist_ok and key in cls._MODULE_MAPPING and cls._MODULE_MAPPING[key] is not factory:
            raise ValueError(f"Module '{key}: {fetch_factory_name(factory)}' is already registered.")
        cls._MODULE_MAPPING[key] = factory
    
    @classmethod
    def unregister(cls, strategy_name: str) -> None:
        cls._MODULE_MAPPING.pop(strategy_name, None)
        
    @classmethod
    def available_modules(cls) -> tuple[str, ...]:
        return tuple(sorted(k for k in cls._MODULE_MAPPING if k != cls._DEFAULT_KEY))