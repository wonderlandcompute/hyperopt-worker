import os
import sys
import yaml
from pathlib import Path

import grpc

from hyperoptWorker.wonderland_pb2_grpc import WonderlandStub


def new_client():
    default_path = os.environ.get("WONDERCOMPUTECONFIG")
    if default_path is None:
        raise Exception("WONDERCOMPUTECONFIG environment variable wasn't set")
    return new_client_from_path(default_path)


def new_client_from_path(config_path):
    config = load_config(config_path)
    creds = load_credentials(config)
    channel = grpc.secure_channel(
        config.get("connect_to"),
        creds,
        options=(
            ('grpc.max_send_message_length', 1024 * 1024 * 1024),
            ('grpc.max_receive_message_length', 1024 * 1024 * 1024),
        )
    )
    return WonderlandStub(channel)


def load_config(config_path):
    if not os.path.exists(config_path):
        raise Exception("Config file `{}` does not exist".format(config_path))

    with open(config_path) as config_f:
        return yaml.load(config_f)


def load_credentials(config):
    ca_cert = Path(config.get("ca_cert")).expanduser()
    client_key = Path(config.get("client_key")).expanduser()
    client_cert = Path(config.get("client_cert")).expanduser()
    path_ok = [
        ca_cert.exists(),
        client_key.exists(),
        client_cert.exists(),
    ]
    if not all(path_ok):
        raise ValueError("One of credentials files does not exist")

    credentials = grpc.ssl_channel_credentials(
        root_certificates=ca_cert.read_bytes(),
        private_key=client_key.read_bytes(),
        certificate_chain=client_cert.read_bytes()
    )

    return credentials


def check_jobs_equal(a, b):
    return (a.project == b.project) and (a.id == b.id) and (a.status == b.status) and (
        a.metadata == b.metadata) and (a.kind == b.kind) and (a.output == b.output) and (a.input == b.input)


def logbar(current, total):
    done = int(50.0 * current / total)
    sys.stdout.write("\r[%s%s]\n" % ('=' * done, ' ' * (50 - done)))
    if current == total:
        sys.stdout.write("\n")
    sys.stdout.flush()

