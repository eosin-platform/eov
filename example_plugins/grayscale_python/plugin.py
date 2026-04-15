"""Grayscale viewport filter plugin for EOV (Python gRPC).

This plugin connects to the EOV extension host via gRPC, registers a
CPU-side viewport filter, and converts rendered frames to grayscale. The host
calls ApplyFilterCpuStream when the filter is enabled.

Requirements (installed in the plugin venv):
    grpcio grpcio-tools numpy

The EOV_EXTENSION_HOST environment variable is set by the host and contains
the gRPC endpoint, e.g. "grpc://localhost:50051".
"""

import os
import sys
import signal
import struct
import threading

# ---------------------------------------------------------------------------
# gRPC generated stubs are produced from proto/eov_extension.proto.
# For simplicity this plugin uses the raw grpcio API to avoid requiring
# a build step with grpc_tools.protoc.
# ---------------------------------------------------------------------------

try:
    import grpc
    from grpc import StatusCode
except ImportError:
    print("[grayscale_python] grpcio not installed; gRPC filter unavailable.", file=sys.stderr)
    sys.exit(1)

try:
    import numpy as np
except ImportError:
    np = None

# ---------------------------------------------------------------------------
# Minimal protobuf serialization helpers (avoids grpc_tools codegen)
#
# We only need a handful of message types so we encode them by hand using the
# protobuf wire format. Field numbers are taken from eov_extension.proto.
# ---------------------------------------------------------------------------

def _encode_varint(value):
    """Encode an unsigned integer as a protobuf varint."""
    out = bytearray()
    while value > 0x7F:
        out.append((value & 0x7F) | 0x80)
        value >>= 7
    out.append(value & 0x7F)
    return bytes(out)


def _encode_string_field(field_number, value):
    tag = _encode_varint((field_number << 3) | 2)
    data = value.encode("utf-8") if isinstance(value, str) else value
    return tag + _encode_varint(len(data)) + data


def _encode_bool_field(field_number, value):
    tag = _encode_varint((field_number << 3) | 0)
    return tag + _encode_varint(1 if value else 0)


def _encode_uint32_field(field_number, value):
    tag = _encode_varint((field_number << 3) | 0)
    return tag + _encode_varint(value)


def _decode_varint(data, pos):
    result = 0
    shift = 0
    while True:
        b = data[pos]
        result |= (b & 0x7F) << shift
        pos += 1
        if not (b & 0x80):
            break
        shift += 7
    return result, pos


def _decode_string_field(data, pos):
    length, pos = _decode_varint(data, pos)
    return data[pos:pos + length], pos + length


def _parse_register_response(data):
    """Parse RegisterFilterResponse -> filter_id (string)."""
    pos = 0
    filter_id = ""
    while pos < len(data):
        tag, pos = _decode_varint(data, pos)
        field_number = tag >> 3
        wire_type = tag & 0x07
        if wire_type == 2:
            value, pos = _decode_string_field(data, pos)
            if field_number == 1:
                filter_id = value.decode("utf-8")
        elif wire_type == 0:
            _, pos = _decode_varint(data, pos)
    return filter_id


# ---------------------------------------------------------------------------
# gRPC filter logic
# ---------------------------------------------------------------------------

def _apply_grayscale(rgba_bytes, width, height):
    """Convert RGBA bytes to grayscale in-place (luminance) and return bytes."""
    length = width * height * 4
    if np is not None:
        arr = np.frombuffer(rgba_bytes, dtype=np.uint8).copy().reshape(-1, 4)
        lum = (0.299 * arr[:, 0] + 0.587 * arr[:, 1] + 0.114 * arr[:, 2]).astype(np.uint8)
        arr[:, 0] = lum
        arr[:, 1] = lum
        arr[:, 2] = lum
        return arr.tobytes()
    else:
        buf = bytearray(rgba_bytes)
        for i in range(0, length, 4):
            r, g, b = buf[i], buf[i + 1], buf[i + 2]
            lum = int(0.299 * r + 0.587 * g + 0.114 * b)
            buf[i] = lum
            buf[i + 1] = lum
            buf[i + 2] = lum
        return bytes(buf)


def _parse_apply_request(data):
    """Parse ApplyFilterCpuRequest -> (filter_id, width, height, rgba_data)."""
    pos = 0
    filter_id = ""
    width = 0
    height = 0
    rgba_data = b""
    while pos < len(data):
        tag, pos = _decode_varint(data, pos)
        field_number = tag >> 3
        wire_type = tag & 0x07
        if wire_type == 2:
            value, pos = _decode_string_field(data, pos)
            if field_number == 1:
                filter_id = value.decode("utf-8")
            elif field_number == 4:
                rgba_data = value
        elif wire_type == 0:
            value, pos = _decode_varint(data, pos)
            if field_number == 2:
                width = value
            elif field_number == 3:
                height = value
    return filter_id, width, height, rgba_data


def _parse_apply_response(data):
    """Parse ApplyFilterCpuResponse -> (rgba_data, width, height)."""
    pos = 0
    rgba_data = b""
    width = 0
    height = 0
    while pos < len(data):
        tag, pos = _decode_varint(data, pos)
        field_number = tag >> 3
        wire_type = tag & 0x07
        if wire_type == 2:
            value, pos = _decode_string_field(data, pos)
            if field_number == 1:
                rgba_data = value
        elif wire_type == 0:
            value, pos = _decode_varint(data, pos)
            if field_number == 2:
                width = value
            elif field_number == 3:
                height = value
    return rgba_data, width, height


def _encode_apply_request(filter_id, width, height, rgba_data):
    """Encode ApplyFilterCpuRequest."""
    return (
        _encode_string_field(1, filter_id)
        + _encode_uint32_field(2, width)
        + _encode_uint32_field(3, height)
        + _encode_string_field(4, rgba_data)
    )


def main():
    host = os.environ.get("EOV_EXTENSION_HOST")
    if not host:
        print("[grayscale_python] EOV_EXTENSION_HOST not set; exiting.", file=sys.stderr)
        sys.exit(1)

    # Strip scheme if present ("grpc://host:port" -> "host:port")
    target = host.replace("grpc://", "")

    channel = grpc.insecure_channel(target)

    # RegisterFilter
    register_req = (
        _encode_string_field(1, "Grayscale (Python)")
        + _encode_bool_field(2, True)   # supports_cpu
        + _encode_bool_field(3, False)  # supports_gpu
    )
    response_data = channel.unary_unary(
        "/eov.extension.ExtensionHost/RegisterFilter",
        request_serializer=lambda x: x,
        response_deserializer=lambda x: x,
    )(register_req, timeout=10)

    filter_id = _parse_register_response(response_data)
    print(f"[grayscale_python] Registered filter with id={filter_id}")

    # SetFilterEnabled = true
    enable_req = _encode_string_field(1, filter_id) + _encode_bool_field(2, True)
    channel.unary_unary(
        "/eov.extension.ExtensionHost/SetFilterEnabled",
        request_serializer=lambda x: x,
        response_deserializer=lambda x: x,
    )(enable_req, timeout=10)
    print("[grayscale_python] Filter enabled")

    # ApplyFilterCpuStream: bidirectional streaming.
    # The host pushes frames via the response stream (ApplyFilterCpuResponse).
    # We process them and send back via the request stream (ApplyFilterCpuRequest).

    import queue as queue_mod
    processed_queue = queue_mod.Queue()

    def request_iterator():
        """Yield initial handshake, then processed frames."""
        # First message: identify ourselves by filter_id.
        yield _encode_string_field(1, filter_id)

        # Subsequent messages: processed frames pushed by the main loop.
        while True:
            item = processed_queue.get()
            if item is None:
                break
            yield item

    try:
        responses = channel.stream_stream(
            "/eov.extension.ExtensionHost/ApplyFilterCpuStream",
            request_serializer=lambda x: x,
            response_deserializer=lambda x: x,
        )(request_iterator())

        print("[grayscale_python] Stream connected, processing frames...")
        for resp_data in responses:
            rgba_data, width, height = _parse_apply_response(resp_data)
            if width == 0 or height == 0 or not rgba_data:
                continue
            processed = _apply_grayscale(rgba_data, width, height)
            processed_queue.put(
                _encode_apply_request(filter_id, width, height, processed)
            )
    except grpc.RpcError as e:
        print(f"[grayscale_python] gRPC error: {e}", file=sys.stderr)
    except KeyboardInterrupt:
        pass
    finally:
        processed_queue.put(None)  # signal request_iterator to stop

        # Unregister on exit
        unreg_req = _encode_string_field(1, filter_id)
        try:
            channel.unary_unary(
                "/eov.extension.ExtensionHost/UnregisterFilter",
                request_serializer=lambda x: x,
                response_deserializer=lambda x: x,
            )(unreg_req, timeout=5)
            print(f"[grayscale_python] Unregistered filter {filter_id}")
        except Exception:
            pass
        channel.close()


if __name__ == "__main__":
    main()
