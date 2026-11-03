# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Import build helpers once without importing the framework-dependent business package."""

import importlib
import importlib.util
from pathlib import Path
import sys
from types import ModuleType

PACKAGE = "_hyper_multicore_build"


def load_module(name: str) -> ModuleType:
    """Keep one module identity for entrypoints, packaging and tests."""
    folder = Path(__file__).resolve().parent
    if PACKAGE not in sys.modules:
        _load(PACKAGE, folder / "__init__.py", [str(folder)])
        sys.modules[PACKAGE + "._bootstrap"] = sys.modules[__name__]
    if name == "native_manifest":
        fullname = PACKAGE + ".native_manifest"
        if fullname not in sys.modules:
            _load(fullname, folder.parent / "_native_manifest.py")
    return importlib.import_module(PACKAGE + "." + name)


def _load(name, path, search=None):
    spec = importlib.util.spec_from_file_location(name, path, submodule_search_locations=search)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module
