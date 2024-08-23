import struct


def encode_semver(major, minor, patch) -> int:
    # https://onnx.ai/onnx/repo-docs/Versioning.html#serializing-semver-version-numbers-in-protobuf
    major_bytes = struct.pack(">H", major)
    minor_bytes = struct.pack(">H", minor)
    patch_bytes = struct.pack(">I", patch)

    semver_bytes = major_bytes + minor_bytes + patch_bytes
    (semver_int,) = struct.unpack(">Q", semver_bytes)

    return semver_int
