import os
from typing import Optional

from dunamai import Version, bump_version, serialize_pep440

if "CI_MERGE_REQUEST_IID" in os.environ:
    mr_version: Optional[str] = os.environ["CI_MERGE_REQUEST_IID"]
else:
    mr_version = None

build_official = "BUILD_OFFICIAL" in os.environ

v = Version.from_git(pattern=r"^(?P<base>\d+\.\d+\.\d+)$")
if v.distance == 0:
    out = serialize_pep440(v.base, v.stage, v.revision)
else:
    if build_official:
        out = serialize_pep440(bump_version(v.base), None, None, dev=v.distance)
    elif mr_version is not None:
        out = serialize_pep440(bump_version(v.base), None, None, dev=v.distance, metadata=[f"mr{mr_version}"])
    else:
        out = serialize_pep440(bump_version(v.base), None, None, dev=v.distance, metadata=[f"g{v.commit}"])

print(out)
