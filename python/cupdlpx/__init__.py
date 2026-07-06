# Copyright 2025 Haihao Lu
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import platform

# Windows only: register CUDA bin for dependent DLL loading.
if platform.system() == "Windows":
    cuda_path = os.environ.get("CUDA_PATH")
    if cuda_path:
        bin_path = os.path.join(cuda_path, "bin")
        if os.path.isdir(bin_path):
            os.add_dll_directory(bin_path)

from .model import Model, read
from . import PDLP

__all__ = ["Model", "read"]

# versioning
from importlib.metadata import version, PackageNotFoundError
# get version from package metadata (toml file)
try:
    __version__ = version("cupdlpx")
except PackageNotFoundError:
    __version__ = "0.0.0"
