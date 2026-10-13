import importlib
import pkgutil

import pytest

import resilens

SKIP = {"resilens.ui.app"}  # needs a running Streamlit session


def all_modules() -> list[str]:
    return [m.name for m in pkgutil.walk_packages(resilens.__path__, prefix="resilens.") if m.name not in SKIP]


@pytest.mark.parametrize("module", all_modules())
def test_module_imports(module: str) -> None:
    importlib.import_module(module)
