# Copyright 2026 Huawei Technologies Co., Ltd.
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
"""Set the wheel's vendor path before TorchNpu snapshots its search paths."""

import os


def _configure_vendor() -> None:
    site_packages = os.path.dirname(os.path.realpath(__file__))
    vendor_name = "hyper_multicore_nn"
    vendor = os.path.join(site_packages, "hyper_multicore/lib/vendors", vendor_name)
    library = "op_api/lib/libcust_opapi.so"
    if not os.path.isfile(os.path.join(vendor, library)):
        return

    variable = "ASCEND_CUSTOM_OPP_PATH"
    configured = os.environ.get(variable, "")
    for entry in filter(None, configured.split(os.pathsep)):
        active_vendor = os.path.realpath(entry)
        if os.path.basename(active_vendor) == vendor_name and os.path.isfile(os.path.join(active_vendor, library)):
            # An explicitly activated checkout or an inherited wheel owns this process.
            return
    os.environ[variable] = os.pathsep.join(filter(None, (vendor, configured)))


_configure_vendor()
