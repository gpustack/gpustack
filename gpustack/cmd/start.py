import argparse
import asyncio
import json
import logging
import multiprocessing
import os
import sys
from typing import Any, Dict

import yaml

from gpustack import __version__, __git_commit__
from gpustack.config.config import set_global_config
from gpustack.logging import setup_logging
from gpustack.utils.envs import get_gpustack_env, get_gpustack_env_bool
from gpustack.worker.worker import Worker
from gpustack.config import Config
from gpustack.server.server import Server
from gpustack.gateway import initialize_gateway


logger = logging.getLogger(__name__)


class OptionalBoolAction(argparse.Action):
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        if nargs is not None:
            raise ValueError("nargs not allowed")
        super(OptionalBoolAction, self).__init__(
            option_strings, dest, nargs=0, **kwargs
        )
        self.default = kwargs.get("default")

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, True)


def setup_start_cmd(subparsers: argparse._SubParsersAction):
    parser_server: argparse.ArgumentParser = subparsers.add_parser(
        "start",
        help="Run GPUStack server or worker.",
        description="Run GPUStack server or worker.",
    )
    start_cmd_options(parser_server)


def start_cmd_options(parser_server: argparse.ArgumentParser):
    common_group = parser_server.add_argument_group("Common settings")
    common_group.add_argument(
        "--advertise-address",
        type=str,
        help="The IP address to expose for external access. If not set, the system will auto-detect a suitable local IP address.",
        default=get_gpustack_env("ADVERTISE_ADDRESS"),
    )
    common_group.add_argument(
        "--port",
        type=int,
        help="Port to bind the server to.",
        default=get_gpustack_env("PORT"),
    )
    common_group.add_argument(
        "--tls-port",
        type=int,
        help="Port to bind the TLS server to.",
        default=get_gpustack_env("TLS_PORT"),
    )
    common_group.add_argument(
        "--api-port",
        type=int,
        help="Port to bind the GPUStack API server to.",
        default=get_gpustack_env("API_PORT"),
    )
    common_group.add_argument(
        "--config-file",
        type=str,
        help="Path to the YAML config file.",
        default=get_gpustack_env("CONFIG_FILE"),
    )
    common_group.add_argument(
        "-d",
        "--debug",
        action=OptionalBoolAction,
        help="Enable debug mode.",
        default=get_gpustack_env_bool("DEBUG"),
    )
    common_group.add_argument(
        "--data-dir",
        type=str,
        help="Directory to store data. The default is OS specific.",
        default=get_gpustack_env("DATA_DIR"),
    )
    common_group.add_argument(
        "--cache-dir",
        type=str,
        help="Directory to store cache (e.g., model files). The default is <data-dir>/cache.",
        default=get_gpustack_env("CACHE_DIR"),
    )
    common_group.add_argument(
        "--bin-dir",
        type=str,
        help="Directory to store additional binaries, e.g., versioned backend executables.",
        default=get_gpustack_env("BIN_DIR"),
    )
    common_group.add_argument(
        "--pipx-path",
        type=str,
        help="Path to the pipx executable, used to install versioned backends.",
        default=get_gpustack_env("PIPX_PATH"),
    )
    common_group.add_argument(
        "--huggingface-token",
        type=str,
        help="User Access Token to authenticate to the Hugging Face Hub.",
        default=os.getenv("HF_TOKEN"),
    )
    common_group.add_argument(
        "--system-default-container-registry",
        type=str,
        help="Default container registry for GPUStack to pull system and inference images. The default is 'docker.io'.",
        default=get_gpustack_env("SYSTEM_DEFAULT_CONTAINER_REGISTRY"),
    )
    common_group.add_argument(
        "--image-name-override",
        type=str,
        help="Override the default image name for the GPUStack container.",
        default=get_gpustack_env("IMAGE_NAME_OVERRIDE"),
    )
    common_group.add_argument(
        "--image-repo",
        type=str,
        help="Override the default image repository gpustack/gpustack for the GPUStack container.",
        default=get_gpustack_env("IMAGE_REPO"),
    )
    common_group.add_argument(
        "--benchmark-image-repo",
        type=str,
        help="Override the default benchmark image repository gpustack/benchmark-runner for the GPUStack benchmark container.",
        default=get_gpustack_env("BENCHMARK_IMAGE_REPO"),
    )
    common_group.add_argument(
        "--gateway-mode",
        type=str,
        help="Gateway running mode. Options: embedded, in-cluster, external, disabled, or auto (default).",
        default=get_gpustack_env("GATEWAY_MODE"),
    )
    common_group.add_argument(
        "--gateway-kubeconfig",
        type=str,
        help="Path to the kubeconfig file for gatway. Only useful for external gateway-mode.",
        default=get_gpustack_env("GATEWAY_KUBECONFIG"),
    )
    common_group.add_argument(
        "--gateway-namespace",
        type=str,
        help="The namespace where the gateway component is deployed.",
        default=get_gpustack_env("GATEWAY_NAMESPACE"),
    )
    common_group.add_argument(
        "--service-discovery-name",
        type=str,
        help="the name of the service discovery service in DNS. Only useful when deployed in Kubernetes with service discovery.",
        default=get_gpustack_env("SERVICE_DISCOVERY_NAME"),
    )
    common_group.add_argument(
        "--namespace",
        type=str,
        help="Kubernetes namespace for GPUStack to deploy gateway routing rules and model instances.",
        default=os.getenv("POD_NAMESPACE"),
    )

    server_group = parser_server.add_argument_group("Server settings")

    # Database settings
    server_group.add_argument(
        "--database-port",
        type=int,
        help="Port of the database. Example: 5432 for PostgreSQL.",
        default=get_gpustack_env("DATABASE_PORT"),
    )
    server_group.add_argument(
        "--metrics-port",
        type=int,
        help="Port to expose server metrics.",
        default=get_gpustack_env("METRICS_PORT"),
    )
    server_group.add_argument(
        "--database-url",
        type=str,
        help="URL of the database. Example: postgresql://user:password@hostname:port/db_name.",
        default=get_gpustack_env("DATABASE_URL"),
    )

    # Embedded worker settings
    server_group.add_argument(
        "--disable-worker",
        action=OptionalBoolAction,
        help="(DEPRECATED) Disable the embedded worker for the GPUStack server. New installations will not have the embedded worker by default. Use '--enable-worker' to enable the embedded worker if needed. If neither flag is set, for backward compatibility, the embedded worker will be enabled by default for legacy installations prior to v2.0.1.",
        default=get_gpustack_env_bool("DISABLE_WORKER"),
    )
    server_group.add_argument(
        "--enable-worker",
        action=OptionalBoolAction,
        help="Enable the embedded worker for the GPUStack server.",
        default=get_gpustack_env_bool("ENABLE_WORKER"),
    )

    # Server settings
    server_group.add_argument(
        "--disable-metrics",
        action=OptionalBoolAction,
        help="Disable server metrics.",
        default=get_gpustack_env_bool(
            "DISABLE_METRICS",
        ),
    )
    server_group.add_argument(
        "--bootstrap-password",
        type=str,
        help="Initial password for the default admin user. Random by default.",
        default=get_gpustack_env("BOOTSTRAP_PASSWORD"),
    )
    server_group.add_argument(
        "--ssl-keyfile",
        type=str,
        help="Path to the SSL key file.",
        default=get_gpustack_env("SSL_KEYFILE"),
    )
    server_group.add_argument(
        "--ssl-certfile",
        type=str,
        help="Path to the SSL certificate file.",
        default=get_gpustack_env("SSL_CERTFILE"),
    )
    server_group.add_argument(
        "--force-auth-localhost",
        action=OptionalBoolAction,
        help="Force authentication for requests originating from localhost (127.0.0.1)."
        "When set to True, all requests from localhost will require authentication.",
        default=get_gpustack_env_bool("FORCE_AUTH_LOCALHOST"),
    )
    server_group.add_argument(
        "--disable-update-check",
        action=OptionalBoolAction,
        help="Disable update check.",
        default=get_gpustack_env_bool("DISABLE_UPDATE_CHECK"),
    )
    server_group.add_argument(
        "--disable-openapi-docs",
        action=OptionalBoolAction,
        help="Disable autogenerated OpenAPI documentation endpoints (Swagger UI at /docs, ReDoc at /redoc, and the raw spec at /openapi.json).",
        default=get_gpustack_env_bool("DISABLE_OPENAPI_DOCS"),
    )
    server_group.add_argument(
        "--update-check-url",
        type=str,
        help=argparse.SUPPRESS,
        default=get_gpustack_env("UPDATE_CHECK_URL"),
    )
    server_group.add_argument(
        "--model-catalog-file",
        type=str,
        help="Path or URL to the model catalog file.",
        default=get_gpustack_env("MODEL_CATALOG_FILE"),
    )
    server_group.add_argument(
        "--server-external-url",
        type=str,
        help="External URL of the server. Should be set if the server is behind a reverse proxy.",
        default=get_gpustack_env("SERVER_EXTERNAL_URL"),
    )
    server_group.add_argument(
        "--gateway-concurrency",
        type=int,
        help="Number of concurrent connections for the embedded gateway. The default is 16.",
        default=get_gpustack_env("GATEWAY_CONCURRENCY"),
    )

    # Observability settings
    server_group.add_argument(
        "--disable-builtin-observability",
        action=OptionalBoolAction,
        help="Disable embedded Grafana and Prometheus services.",
        default=get_gpustack_env_bool("DISABLE_BUILTIN_OBSERVABILITY"),
    )
    server_group.add_argument(
        "--grafana-url",
        type=str,
        help="Grafana base URL for dashboard redirects and proxying. Must be browser-reachable (not a container-only hostname). If set, embedded Grafana and Prometheus will be disabled. Only required for external Grafana.",
        default=get_gpustack_env("GRAFANA_URL"),
    )
    server_group.add_argument(
        "--grafana-worker-dashboard-uid",
        type=str,
        help="Grafana dashboard UID for worker dashboard redirects.",
        default=get_gpustack_env("GRAFANA_WORKER_DASHBOARD_UID"),
    )
    server_group.add_argument(
        "--grafana-model-dashboard-uid",
        type=str,
        help="Grafana dashboard UID for model dashboard redirects.",
        default=get_gpustack_env("GRAFANA_MODEL_DASHBOARD_UID"),
    )

    # CORS settings
    server_group.add_argument(
        "--enable-cors",
        action=OptionalBoolAction,
        help="Enable Cross-Origin Resource Sharing (CORS) on the server.",
        default=get_gpustack_env_bool("ENABLE_CORS"),
    )
    server_group.add_argument(
        "--allow-credentials",
        action=OptionalBoolAction,
        help="Allow cookies and credentials in cross-origin requests.",
        default=get_gpustack_env_bool("ALLOW_CREDENTIALS"),
    )
    server_group.add_argument(
        "--allow-origins",
        action='append',
        help='Origins allowed for cross-origin requests. Specify the flag multiple times for multiple origins. Example: --allow-origins https://example.com --allow-origins https://api.example.com. Default: ["*"] (all origins allowed).',
    )
    server_group.add_argument(
        "--allow-methods",
        action='append',
        help='HTTP methods allowed in cross-origin requests. Specify the flag multiple times for multiple methods. Example: --allow-methods GET --allow-methods POST. Default: ["GET", "POST"].',
    )
    server_group.add_argument(
        "--allow-headers",
        action='append',
        help='HTTP request headers allowed in cross-origin requests. Specify the flag multiple times for multiple headers. Example: --allow-headers Authorization --allow-headers Content-Type. Default: ["Authorization", "Content-Type"].',
    )

    # OIDC settings
    server_group.add_argument(
        "--oidc-issuer",
        type=str,
        help="The issuer URL of the OIDC provider. OIDC discovery under `<issuer>/.well-known/openid-configuration` will be used to discover the OIDC configuration.",
        default=get_gpustack_env("OIDC_ISSUER"),
    )
    server_group.add_argument(
        "--oidc-client-id",
        type=str,
        help="OIDC client ID.",
        default=get_gpustack_env("OIDC_CLIENT_ID"),
    )
    server_group.add_argument(
        "--oidc-client-secret",
        type=str,
        help="OIDC client secret.",
        default=get_gpustack_env("OIDC_CLIENT_SECRET"),
    )
    server_group.add_argument(
        "--oidc-redirect-uri",
        type=str,
        help="The redirect URI configured in your OIDC application. This must be set to `<server-url>/auth/oidc/callback`.",
        default=get_gpustack_env("OIDC_REDIRECT_URI"),
    )
    server_group.add_argument(
        "--oidc-skip-userinfo",
        action=OptionalBoolAction,
        help="Skip using the UserInfo endpoint and retrieve user details from the ID token.",
        default=get_gpustack_env_bool("OIDC_SKIP_USERINFO"),
    )
    server_group.add_argument(
        "--oidc-use-userinfo",
        action=OptionalBoolAction,
        help="[Deprecated] Use the UserInfo endpoint to fetch user details after authentication.",
        default=get_gpustack_env_bool("OIDC_USE_USERINFO"),
    )

    # SAML settings
    server_group.add_argument(
        "--saml-idp-server-url",
        type=str,
        help="SAML IdP server URL.",
        default=get_gpustack_env("SAML_IDP_SERVER_URL"),
    )
    server_group.add_argument(
        "--saml-idp-logout-url",
        type=str,
        help="SAML IdP Single Logout endpoint URL.",
        default=get_gpustack_env("SAML_IDP_LOGOUT_URL"),
    )
    server_group.add_argument(
        "--saml-idp-entity-id",
        type=str,
        help="SAML IdP entity ID.",
        default=get_gpustack_env("SAML_IDP_ENTITY_ID"),
    )
    server_group.add_argument(
        "--saml-idp-x509-cert",
        type=str,
        help="SAML IdP X.509 certificate.",
        default=get_gpustack_env("SAML_IDP_X509_CERT"),
    )
    server_group.add_argument(
        "--saml-sp-entity-id",
        type=str,
        help="SAML SP entity ID.",
        default=get_gpustack_env("SAML_SP_ENTITY_ID"),
    )
    server_group.add_argument(
        "--saml-sp-acs-url",
        type=str,
        help="SAML SP Assertion Consumer Service(ACS) URL. It should be set to `<server-url>/auth/saml/callback`.",
        default=get_gpustack_env("SAML_SP_ACS_URL"),
    )
    server_group.add_argument(
        "--saml-sp-slo-url",
        type=str,
        help="SAML SP Single Logout Service URL. It can be set to `<server-url>/auth/saml/logout/callback` if you need to receive LogoutResponse.",
        default=get_gpustack_env("SAML_SP_SLO_URL"),
    )
    server_group.add_argument(
        "--saml-sp-x509-cert",
        type=str,
        help="SAML SP X.509 certificate.",
        default=get_gpustack_env("SAML_SP_X509_CERT"),
    )
    server_group.add_argument(
        "--saml-sp-private-key",
        type=str,
        help="SAML SP private key.",
        default=get_gpustack_env("SAML_SP_PRIVATE_KEY"),
    )
    server_group.add_argument(
        "--saml-sp-attribute-prefix",
        type=str,
        help="SAML Service Provider attribute prefix, which is used for fetching the attributes that are specified by --external-auth-*. e.g., 'http://schemas.auth0.com/'.",
        default=get_gpustack_env("SAML_SP_ATTRIBUTE_PREFIX"),
    )
    server_group.add_argument(
        "--saml-security",
        type=str,
        help="SAML security settings in JSON.",
        default=get_gpustack_env("SAML_SECURITY"),
    )

    # External Authentication settings
    server_group.add_argument(
        "--external-auth-name",
        type=str,
        help="Mapping of external authentication user information to username, e.g., 'preferred_username'. For SAML, you must configure the full attribute name like 'http://schemas.xmlsoap.org/ws/2005/05/identity/claims/emailaddress' or simplify with 'emailaddress' by '--saml-sp-attribute-prefix'.",
        default=get_gpustack_env("EXTERNAL_AUTH_NAME"),
    )
    server_group.add_argument(
        "--external-auth-full-name",
        type=str,
        help="Mapping of external authentication user information to user's full name. Multiple elements can be combined, e.g., 'name' or 'firstName+lastName'.  For SAML, you must configure the full attribute name like 'http://schemas.xmlsoap.org/ws/2005/05/identity/claims/name' or simplify with 'name' by '--saml-sp-attribute-prefix'.",
        default=get_gpustack_env("EXTERNAL_AUTH_FULL_NAME"),
    )
    server_group.add_argument(
        "--external-auth-avatar-url",
        type=str,
        help="Mapping of external authentication user information to user's avatar URL. e.g.,'picture'. For SAML, you must configure the full attribute name like 'http://schemas.auth0.com/picture' or simplify with 'picture' by '--saml-sp-attribute-prefix'.",
        default=get_gpustack_env("EXTERNAL_AUTH_AVATAR_URL"),
    )
    server_group.add_argument(
        "--external-auth-default-inactive",
        action=OptionalBoolAction,
        help="Set newly created externally authenticated users inactive by default.",
        default=get_gpustack_env_bool("EXTERNAL_AUTH_DEFAULT_INACTIVE"),
    )
    server_group.add_argument(
        "--external-auth-post-logout-redirect-key",
        type=str,
        help="Generic key for post-logout redirection across IdPs.",
        default=get_gpustack_env("EXTERNAL_AUTH_POST_LOGOUT_REDIRECT_KEY"),
    )

    worker_group = parser_server.add_argument_group("Worker settings")
    worker_group.add_argument(
        "-t",
        "--token",
        type=str,
        help="Shared secret used to add a worker.",
        default=get_gpustack_env("TOKEN"),
    )
    worker_group.add_argument(
        "-s",
        "--server-url",
        type=str,
        help="Server to connect to.",
        default=get_gpustack_env("SERVER_URL"),
    )
    worker_group.add_argument(
        "--worker-ip",
        type=str,
        help="IP address of the worker node. Auto-detected by default.",
        default=get_gpustack_env("WORKER_IP"),
    )
    worker_group.add_argument(
        "--worker-ifname",
        type=str,
        help="Network interface name of the worker node. Auto-detected by default.",
        default=get_gpustack_env("WORKER_IFNAME"),
    )
    worker_group.add_argument(
        "--worker-name",
        type=str,
        help="Name of the worker node. Use the hostname by default.",
        default=get_gpustack_env("WORKER_NAME"),
    )
    worker_group.add_argument(
        "--worker-port",
        type=int,
        help="Port to bind the worker to.",
        default=get_gpustack_env("WORKER_PORT"),
    )
    worker_group.add_argument(
        "--service-port-range",
        type=str,
        help="Port range for inference services, specified as a string in the form 'N1-N2'. Both ends of the range are inclusive. The default is '40000-40063'.",
        default=get_gpustack_env("SERVICE_PORT_RANGE"),
    )
    worker_group.add_argument(
        "--ray-port-range",
        type=str,
        help="Port range for Ray services(vLLM distributed deployment using), specified as a string in the form 'N1-N2'. Both ends of the range are inclusive. The default is '41000-41999'.",
        default=get_gpustack_env("RAY_PORT_RANGE"),
    )
    worker_group.add_argument(
        "--benchmark-max-duration-seconds",
        type=int,
        help="Max duration for a benchmark before timeout. Disabled when unset.",
        default=get_gpustack_env("BENCHMARK_MAX_DURATION_SECONDS"),
    )
    worker_group.add_argument(
        "--disable-worker-metrics",
        action=OptionalBoolAction,
        help="Disable worker metrics.",
        default=get_gpustack_env_bool(
            "DISABLE_WORKER_METRICS",
        ),
    )
    worker_group.add_argument(
        "--worker-metrics-port",
        type=int,
        help="Port to expose worker metrics.",
        default=get_gpustack_env("WORKER_METRICS_PORT"),
    )
    worker_group.add_argument(
        "--log-dir",
        type=str,
        help="Directory to store logs.",
        default=get_gpustack_env("LOG_DIR"),
    )
    worker_group.add_argument(
        "--benchmark-dir",
        type=str,
        help="Directory to store benchmark results.",
        default=get_gpustack_env("BENCHMARK_DIR"),
    )
    worker_group.add_argument(
        "--system-reserved",
        type=json.loads,
        help="The system reserves resources during scheduling, measured in GiB. \
        Where RAM is reserved per worker, and VRAM is reserved per GPU device. \
        By default, no resources are reserved. \
        Example: '{\"ram\": 2, \"vram\": 1}' or '{\"memory\": 2, \"gpu_memory\": 1}', \
        Note: The 'memory' and 'gpu_memory' keys are deprecated and will be removed in future releases.",
        default=get_gpustack_env("SYSTEM_RESERVED"),
    )
    worker_group.add_argument(
        "--tools-download-base-url",
        type=str,
        help="Base URL to download dependency tools.",
        default=get_gpustack_env("TOOLS_DOWNLOAD_BASE_URL"),
    )
    worker_group.add_argument(
        "--enable-hf-transfer",
        action=OptionalBoolAction,
        help="Enable faster downloads from the Hugging Face Hub using hf_transfer.",
        default=os.getenv("HF_HUB_ENABLE_HF_TRANSFER"),
    )
    worker_group.add_argument(
        "--enable-hf-xet",
        action=OptionalBoolAction,
        help="Enable downloading model files using Hugging Face Xet.",
    )
    worker_group.add_argument(
        "--proxy-mode",
        type=str,
        help="Proxy mode for server accessing model instances: "
        "direct (server connects directly) or worker (via worker proxy). "
        "Default value is direct for embedded worker, and worker for standalone worker.",
    )

    parser_server.set_defaults(func=run)


def run(args: argparse.Namespace):
    try:
        cfg = parse_args(args)
        setup_logging(cfg.debug)
        debug_env_info()
        set_third_party_env(cfg=cfg)
        set_ulimit()
        start_tracemalloc_if_debug(cfg)
        initialize_gateway(cfg)
        multiprocessing.set_start_method('spawn')

        logger.info(f"GPUStack version: {__version__} ({__git_commit__})")

        if cfg.server_url:
            run_worker(cfg)
        else:
            run_server(cfg)
    except Exception as e:
        logger.fatal(e)


def run_server(cfg: Config):
    worker = Worker(cfg)

    server = Server(
        config=cfg, worker_process=multiprocessing.Process(target=worker.start)
    )

    try:
        asyncio.run(server.start())
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    except Exception as e:
        logger.exception(f"Error running server: {e}")
        # logger.error(f"Error running server: {e}")
    finally:
        logger.info("Server has shut down.")


def run_worker(cfg: Config):
    worker = Worker(cfg)

    worker.start()


def load_config_from_yaml(yaml_file: str) -> Dict[str, Any]:
    with open(yaml_file, "r") as file:
        return yaml.safe_load(file)


def parse_args(args: argparse.Namespace) -> Config:
    config_data = {}
    if args.config_file:
        config_data.update(load_config_from_yaml(args.config_file))

    # CLI args have higher priority than config file
    set_common_options(args, config_data)
    set_server_options(args, config_data)
    set_worker_options(args, config_data)

    try:
        cfg = Config(**config_data)
    except Exception as e:
        raise Exception(f"Config error: {e}")

    set_global_config(cfg)
    return cfg


def set_config_option(args, config_data: dict, option_name: str):
    option_value = getattr(args, option_name, None)
    if option_value is not None:
        config_data[option_name] = option_value


def set_common_options(args, config_data: dict):
    options = [
        "debug",
        "data_dir",
        "cache_dir",
        "bin_dir",
        "pipx_path",
        "huggingface_token",
        "system_default_container_registry",
        "image_name_override",
        "image_repo",
        "benchmark_image_repo",
        "advertise_address",
        "port",
        "tls_port",
        "api_port",
        "gateway_mode",
        "gateway_kubeconfig",
        "gateway_namespace",
        "service_discovery_name",
        "namespace",
    ]

    for option in options:
        set_config_option(args, config_data, option)


def set_server_options(args, config_data: dict):
    options = [
        "metrics_port",
        "database_port",
        "disable_metrics",
        "database_url",
        "disable_worker",
        "enable_worker",
        "bootstrap_password",
        "ssl_keyfile",
        "ssl_certfile",
        "force_auth_localhost",
        "disable_update_check",
        "disable_openapi_docs",
        "update_check_url",
        "model_catalog_file",
        "enable_cors",
        "allow_origins",
        "allow_credentials",
        "allow_methods",
        "allow_headers",
        "external_auth_name",
        "external_auth_full_name",
        "external_auth_avatar_url",
        "external_auth_default_inactive",
        "oidc_issuer",
        "oidc_client_id",
        "oidc_client_secret",
        "oidc_redirect_uri",
        "external_auth_post_logout_redirect_key",
        "oidc_skip_userinfo",
        "oidc_use_userinfo",
        "saml_idp_server_url",
        "saml_idp_logout_url",
        "saml_idp_entity_id",
        "saml_idp_x509_cert",
        "saml_sp_entity_id",
        "saml_sp_acs_url",
        "saml_sp_slo_url",
        "saml_sp_x509_cert",
        "saml_sp_private_key",
        "saml_sp_attribute_prefix",
        "saml_security",
        "server_external_url",
        "gateway_concurrency",
        "disable_builtin_observability",
        "grafana_url",
        "grafana_worker_dashboard_uid",
        "grafana_model_dashboard_uid",
    ]

    for option in options:
        set_config_option(args, config_data, option)


def set_worker_options(args, config_data: dict):
    options = [
        "token",
        "server_url",
        "worker_ip",
        "worker_ifname",
        "worker_name",
        "worker_port",
        "disable_worker_metrics",
        "worker_metrics_port",
        "service_port_range",
        "ray_port_range",
        "benchmark_max_duration_seconds",
        "log_dir",
        "benchmark_dir",
        "system_reserved",
        "tools_download_base_url",
        "enable_hf_transfer",
        "enable_hf_xet",
        "proxy_mode",
    ]

    for option in options:
        set_config_option(args, config_data, option)


def debug_env_info():
    hf_endpoint = os.getenv("HF_ENDPOINT")
    if hf_endpoint:
        logger.debug(f"Using HF_ENDPOINT: {hf_endpoint}")


def start_tracemalloc_if_debug(cfg: Config):
    if cfg.debug:
        import tracemalloc

        tracemalloc.start()
        logger.debug("tracemalloc started for memory profiling")


def set_third_party_env(cfg: Config):
    if cfg.enable_hf_transfer:
        # https://huggingface.co/docs/huggingface_hub/guides/download#faster-downloads
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
        logger.debug("set env HF_HUB_ENABLE_HF_TRANSFER=1")

    if not cfg.enable_hf_xet:
        os.environ["HF_HUB_DISABLE_XET"] = "1"
        logger.debug("set env HF_HUB_DISABLE_XET=1")


# Adapted from: https://github.com/vllm-project/vllm/blob/main/vllm/utils.py#L2438
def set_ulimit(target_soft_limit=65535):
    if sys.platform.startswith('win'):
        logger.info("Windows detected, skipping ulimit adjustment.")
        return
    import resource

    resource_type = resource.RLIMIT_NOFILE
    current_soft, current_hard = resource.getrlimit(resource_type)

    if current_soft < target_soft_limit:
        try:
            resource.setrlimit(resource_type, (target_soft_limit, current_hard))
            logger.info(
                f"Increase the ulimit from {current_soft} to {target_soft_limit}."
            )
        except ValueError as e:
            logger.warning(
                f"Failed to set ulimit (nofile): {e}. "
                f"Current soft limit: {current_soft}. "
                "Consider increasing with `ulimit -n`."
            )
