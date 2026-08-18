import hashlib
from typing import Tuple
from urllib.parse import urlparse
from enum import Enum
from typing import (
    ClassVar,
    Optional,
    List,
    Union,
    TYPE_CHECKING,
    Literal,
    Mapping,
    Dict,
    Any,
)
from pydantic import (
    BaseModel,
    ConfigDict,
    field_validator,
    model_validator,
    Field as PydanticField,
)
from sqlalchemy import UniqueConstraint
from sqlmodel import (
    Field,
    Column,
    ForeignKey,
    Integer,
    JSON,
    SQLModel,
    Relationship,
)

from gpustack.mixins import BaseModelMixin
from gpustack.schemas.common import (
    PublicFields,
    ListParams,
    PaginatedList,
    pydantic_column_type,
)

if TYPE_CHECKING:
    from gpustack.schemas.model_routes import ModelRouteTarget


# The ``anthropic-version`` every Claude request carries. Lives here rather than
# in the route module because both paths to a Claude provider need it and they
# are not the same code: the gateway gets it from the provider config, and
# get-models / test-model call the provider directly and set the header
# themselves. Two copies of the value drift into a provider that tests fine from
# the UI and fails in inference, or the reverse.
#
# ai-proxy would default this on its own (``claudeDefaultVersion``, same value
# today), but the default is the plugin's and can move under us on a re-sync. It
# is sent explicitly so the version in effect is the one recorded here.
ANTHROPIC_API_VERSION = "2023-06-01"

# The model-listing path of the OpenAI API, which most providers -- and
# Anthropic -- serve unchanged. Named here rather than repeated per config so a
# route that has to speak the path itself (rather than getting it back from
# ``get_model_url``) references the same string the configs do.
V1_MODELS_URI = "/v1/models"


# The provider types should be synced with higress ai-proxy supported providers
class ModelProviderTypeEnum(str, Enum):
    AI360 = "ai360"
    AZURE = "azure"
    BAICHUAN = "baichuan"
    BAIDU = "baidu"
    BEDROCK = "bedrock"
    CLAUDE = "claude"
    CLOUDFLARE = "cloudflare"
    COHERE = "cohere"
    COZE = "coze"
    DEEPL = "deepl"
    DEEPSEEK = "deepseek"
    DIFY = "dify"
    DOUBAO = "doubao"
    FIREWORKS = "fireworks"
    GALADRIEL = "galadriel"
    GEMINI = "gemini"
    GENERIC = "generic"
    GITHUB = "github"
    GROK = "grok"
    GROQ = "groq"
    HUNYUAN = "hunyuan"
    LONGCAT = "longcat"
    MINIMAX = "minimax"
    MISTRAL = "mistral"
    MOONSHOT = "moonshot"
    OLLAMA = "ollama"
    OPENAI = "openai"
    OPENROUTER = "openrouter"
    QWEN = "qwen"
    SPARK = "spark"
    STEPFUN = "stepfun"
    TOGETHERAI = "together-ai"
    TRITON = "triton"
    YI = "yi"
    ZHIPUAI = "zhipuai"

    # following types are not supported yet
    # For vertex, It has more complex configuration than other providers. Keep it unsupported for now.
    # VERTEX     = "vertex"
    # For vllm, most of the vllm provider functions can be replaced by open-ai compatible provider.
    # VLLM       = "vllm"


class BaseProviderConfig(BaseModel):
    model_config: ConfigDict = {
        "extra": "allow",
    }
    _chat_uri: Optional[str] = "/v1/chat/completions"
    _public_endpoint: Optional[str] = None
    _default_schema = "https"
    _model_uri = None

    def get_base_url(self) -> Optional[str]:
        if self._public_endpoint:
            return f"{self._default_schema}://{self._public_endpoint}"
        return None

    def check_required_fields(self):
        missing_fields = []
        for name, field in self.__class__.model_fields.items():
            schema_extra = field.json_schema_extra or {}
            if schema_extra.get("field_required", False):
                value = getattr(self, name)
                if value is None:
                    missing_fields.append(name)
        if missing_fields:
            raise ValueError(
                f"Missing required fields for provider {self.type}: {', '.join(missing_fields)}"
            )
        return self

    def get_model_url(self) -> Tuple[Optional[str], Optional[str]]:
        base_url = self.get_base_url()
        if base_url:
            base_url = base_url.rstrip("/")
        return base_url, self._model_uri

    def get_chat_url(self) -> Tuple[Optional[str], Optional[str]]:
        base_url = self.get_base_url()
        if base_url:
            base_url = base_url.rstrip("/")
        return base_url, self._chat_uri

    def ai_proxy_derived_fields(self) -> Dict[str, Any]:
        """ai-proxy fields this config implies but does not store.

        For the knobs where our field and the plugin's are the same thing under
        the same name, the dump carries them across on its own. This is for the
        ones where they are not: a single URL of ours that the plugin expects
        split across several of its own fields. Ranked below explicitly set
        values, same as ``_default_override``, so the escape hatch of writing
        the plugin's own field names into the config still wins.
        """
        return {}

    def model_dump_with_default_override(self) -> Dict[str, Any]:
        """Dumps the model, excluding unset fields, and then merges with `_default_override` values.

        This method is used to generate a configuration dictionary for services
        that require certain default values to be present, even if they are not
        explicitly set by the user. User-set values will take precedence over
        the default override values.

        The `_default_override` attribute should be a dictionary defined on the
        config subclass.
        """
        default_override = getattr(self, "_default_override", {})
        values = {
            **default_override,
            **self.ai_proxy_derived_fields(),
            **self.model_dump(exclude_unset=True, exclude={"type"}),
        }
        return values


class Ai360Config(BaseProviderConfig):
    type: Literal[ModelProviderTypeEnum.AI360]
    _public_endpoint: str = "api.360.cn"


class AzureOpenAIConfig(BaseProviderConfig):
    type: Literal[ModelProviderTypeEnum.AZURE]
    azureServiceUrl: Optional[str] = PydanticField(
        default=None, json_schema_extra={"field_required": True}
    )

    def get_base_url(self) -> Optional[str]:
        return self.azureServiceUrl


class BaichuanConfig(BaseProviderConfig):
    type: Literal[ModelProviderTypeEnum.BAICHUAN]
    _public_endpoint: str = "api.baichuan-ai.com"
    _model_uri = V1_MODELS_URI


class BaiduConfig(BaseProviderConfig):
    type: Literal[ModelProviderTypeEnum.BAIDU]
    _public_endpoint: str = "qianfan.baidubce.com"
    _model_uri = V1_MODELS_URI


class BedrockConfig(BaseProviderConfig):
    type: Literal[ModelProviderTypeEnum.BEDROCK]
    awsAccessKey: Optional[str] = PydanticField(
        default=None, json_schema_extra={"field_required": True}
    )
    awsSecretKey: Optional[str] = PydanticField(
        default=None, json_schema_extra={"field_required": True}
    )
    awsRegion: Optional[str] = PydanticField(
        default=None, json_schema_extra={"field_required": True}
    )
    bedrockAdditionalFields: Optional[dict] = None

    def get_base_url(self):
        return (
            f"{self._default_schema}://bedrock-runtime.{self.awsRegion}.amazonaws.com"
        )


class ClaudeConfig(BaseProviderConfig):
    """Anthropic, or anything that speaks its API.

    ``claudeCustomUrl`` is the Anthropic-side counterpart of
    ``openaiCustomUrl``, but it reaches the plugin differently. ai-proxy's
    claude provider hardcodes ``api.anthropic.com`` and has no per-provider URL
    field, so the custom endpoint is expressed with the generic knobs instead --
    see ``ai_proxy_derived_fields``.
    """

    type: Literal[ModelProviderTypeEnum.CLAUDE]
    claudeVersion: Optional[str] = ANTHROPIC_API_VERSION
    claudeCustomUrl: Optional[str] = None
    _public_endpoint: str = "api.anthropic.com"
    _model_uri = V1_MODELS_URI
    _chat_uri = "/v1/messages"
    # Also in the override, not just as the field default: the field default is
    # dropped by ``exclude_unset``, so a provider stored before this default
    # existed would still send no version at all and fall back to the plugin's.
    # This does add a key to the deployed config of every existing Claude
    # provider, which is the cost not paid for ``protocol``. The difference is
    # what the key decides: the version has to be the same on both paths into a
    # provider and is only ever the value above, while the protocol decides
    # whether requests are translated at all, so writing one is recording a fact
    # and writing the other would be making a choice on the operator's behalf.
    _default_override = {"claudeVersion": ANTHROPIC_API_VERSION}

    @field_validator("claudeCustomUrl")
    @classmethod
    def check_claude_custom_url(cls, value: Optional[str]) -> Optional[str]:
        """An origin, optionally a path prefix, and nothing else.

        The host becomes an McpBridge registry, and a value like
        ``192.168.50.14`` parses as a *path* with no host at all -- which yields
        the base URL ``https://`` and a registry Higress rejects. The scheme
        picks the registry protocol, so ``http`` against a plain-HTTP endpoint
        is the difference between working and a TLS handshake against a server
        that does not speak it.

        Credentials are refused rather than quietly dropped, because the netloc
        travels on as ``providerDomain`` and ai-proxy writes that to
        ``:authority`` verbatim (``OverwriteRequestHostHeader``): a
        ``user:pw@host`` URL would put them in the header of every proxied
        request, and in the access logs of everything on the way. A query or a
        fragment is refused for the plainer reason that only the origin and the
        path are ever read -- accepting them would imply they are sent.
        """
        if value is None:
            return value
        parsed = urlparse(value)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise ValueError(
                "claudeCustomUrl must be an absolute http(s) URL, "
                f"e.g. http://192.168.50.14:8080; got {value!r}"
            )
        if parsed.username or parsed.password:
            raise ValueError(
                "claudeCustomUrl must not carry credentials: they would be sent "
                "as the Host header of every request. Configure the endpoint's "
                f"key as the provider API key instead; got {value!r}"
            )
        if parsed.query or parsed.fragment:
            raise ValueError(
                "claudeCustomUrl takes an origin and an optional path prefix "
                f"only; the query or fragment in {value!r} would be dropped"
            )
        return value

    def _custom_base_path(self) -> str:
        """The path prefix of the custom URL, without its trailing slash.

        Empty when there is none, which is the common case: an Anthropic-
        compatible server usually serves ``/v1/messages`` at the root.
        """
        if not self.claudeCustomUrl:
            return ""
        return urlparse(self.claudeCustomUrl).path.rstrip("/")

    def get_base_url(self) -> Optional[str]:
        if self.claudeCustomUrl:
            parsed = urlparse(self.claudeCustomUrl)
            return f"{parsed.scheme}://{parsed.netloc}"
        return super().get_base_url()

    def get_model_url(self) -> Tuple[Optional[str], Optional[str]]:
        base_url, model_uri = super().get_model_url()
        return base_url, self._prefix_custom_base_path(model_uri)

    def get_chat_url(self) -> Tuple[Optional[str], Optional[str]]:
        base_url, chat_uri = super().get_chat_url()
        return base_url, self._prefix_custom_base_path(chat_uri)

    def _prefix_custom_base_path(self, uri: Optional[str]) -> Optional[str]:
        base_path = self._custom_base_path()
        if not base_path or uri is None:
            return uri
        return f"{base_path}{uri}"

    def ai_proxy_derived_fields(self) -> Dict[str, Any]:
        """``claudeCustomUrl`` as the two knobs ai-proxy actually reads.

        ``providerDomain`` and ``providerBasePath`` are generic: the plugin
        applies them in ``handleRequestHeaders`` *after* the claude provider has
        overwritten the authority with ``api.anthropic.com``, so they win. The
        netloc rather than the hostname, so a non-default port reaches the
        upstream in the authority the way HTTP expects.

        An endpoint mounted at the root sends no ``providerBasePath`` at all
        rather than ``/``. ``applyProviderBasePath`` prepends the value only
        when the path does not already start with it, so ``/`` is a no-op on
        every path there is -- and the shorter config is the one whose diff
        against what is deployed means something.

        ``protocol`` is deliberately not derived here. Unset, the plugin's own
        default applies and the provider is exposed as OpenAI, converted both
        ways; ``protocol: original`` -- set by hand on the config, through the
        ``extra="allow"`` escape hatch -- forwards unconverted instead, keeping
        what has no OpenAI equivalent (``cache_control``, thinking blocks,
        tool-use shapes) at the cost of serving the Anthropic protocol only.
        Which of the two an endpoint should be exposed as is not something its
        URL implies, so it stays the operator's choice.
        """
        if not self.claudeCustomUrl:
            return {}
        derived: Dict[str, Any] = {
            "providerDomain": urlparse(self.claudeCustomUrl).netloc,
        }
        base_path = self._custom_base_path()
        if base_path:
            derived["providerBasePath"] = base_path
        return derived


class CloudflareConfig(BaseProviderConfig):
    type: Literal[ModelProviderTypeEnum.CLOUDFLARE]
    cloudflareAccountId: Optional[str] = PydanticField(
        default=None, json_schema_extra={"field_required": True}
    )

    _public_endpoint: str = "api.cloudflare.com"
    _model_uri = None


class CohereConfig(BaseProviderConfig):
    type: Literal[ModelProviderTypeEnum.COHERE]
    _public_endpoint: str = "api.cohere.com"


class CozeConfig(BaseProviderConfig):
    type: Literal[ModelProviderTypeEnum.COZE]
    _public_endpoint: str = "api.coze.cn"


class DeeplConfig(BaseProviderConfig):
    type: Literal[ModelProviderTypeEnum.DEEPL]
    targetLang: Optional[str] = PydanticField(
        default=None, json_schema_extra={"field_required": True}
    )
    _public_endpoint: str = "api-free.deepl.com"


class DeepseekConfig(BaseProviderConfig):
    type: Literal[ModelProviderTypeEnum.DEEPSEEK]
    _public_endpoint: str = "api.deepseek.com"
    _model_uri = V1_MODELS_URI


class DifyConfig(BaseProviderConfig):
    type: Literal[ModelProviderTypeEnum.DIFY]
    difyApiUrl: Optional[str] = None
    botType: Optional[str] = None
    inputVariable: Optional[str] = None
    outputVariable: Optional[str] = None
    _public_endpoint: str = "api.dify.ai"

    def get_base_url(self) -> Optional[str]:
        if self.difyApiUrl:
            return self.difyApiUrl
        return super().get_base_url()


class DoubaoConfig(BaseProviderConfig):
    type: Literal[ModelProviderTypeEnum.DOUBAO]
    doubaoDomain: Optional[str] = None

    _public_endpoint: str = "ark.cn-beijing.volces.com"
    _model_uri = "/api/v3/models"
    _chat_uri = "/api/v3/chat/completions"

    def get_base_url(self):
        domain = self.doubaoDomain or self._public_endpoint
        return f"{self._default_schema}://{domain}"


class FireworksConfig(BaseProviderConfig):
    type: Literal[ModelProviderTypeEnum.FIREWORKS]
    _public_endpoint: str = "api.fireworks.ai"
    _model_uri = V1_MODELS_URI


class GaladrielConfig(BaseProviderConfig):
    type: Literal[ModelProviderTypeEnum.GALADRIEL]
    _public_endpoint: str = "api.galadriel.com"
    _model_uri = V1_MODELS_URI


class GeminiConfig(BaseProviderConfig):
    type: Literal[ModelProviderTypeEnum.GEMINI]
    geminiSafetySetting: Optional[Mapping[str, str]] = None
    apiVersion: Optional[str] = None
    geminiThinkingBudget: Optional[float] = None
    _public_endpoint: str = "generativelanguage.googleapis.com"
    _default_override = {"apiVersion": "v1beta"}


class GenericConfig(BaseProviderConfig):
    type: Literal[ModelProviderTypeEnum.GENERIC]
    _public_endpoint: str = ""

    def get_base_url(self) -> Optional[str]:
        return None


class GithubConfig(BaseProviderConfig):
    type: Literal[ModelProviderTypeEnum.GITHUB]
    _public_endpoint: str = "models.inference.ai.azure.com"


class GrokConfig(BaseProviderConfig):
    type: Literal[ModelProviderTypeEnum.GROK]
    _public_endpoint: str = "api.x.ai"


class GroqConfig(BaseProviderConfig):
    type: Literal[ModelProviderTypeEnum.GROQ]
    _public_endpoint: str = "api.groq.com"


class HunyuanConfig(BaseProviderConfig):
    type: Literal[ModelProviderTypeEnum.HUNYUAN]
    hunyuanAuthId: Optional[str] = PydanticField(
        default=None, json_schema_extra={"field_required": True}
    )
    hunyuanAuthKey: Optional[str] = PydanticField(
        default=None, json_schema_extra={"field_required": True}
    )
    _public_endpoint: str = "hunyuan.tencentcloudapi.com"


class LongcatConfig(BaseProviderConfig):
    type: Literal[ModelProviderTypeEnum.LONGCAT]
    _public_endpoint: str = "api.longcat.chat"
    _model_uri = V1_MODELS_URI


class MinimaxConfig(BaseProviderConfig):
    type: Literal[ModelProviderTypeEnum.MINIMAX]
    minimaxApiType: Optional[str] = None
    minimaxGroupId: Optional[str] = None
    _public_endpoint: str = "api.minimax.chat"
    _default_override = {"minimaxApiType": "v2"}


class MistralConfig(BaseProviderConfig):
    type: Literal[ModelProviderTypeEnum.MISTRAL]
    _public_endpoint: str = "api.mistral.ai"


class MoonshotConfig(BaseProviderConfig):
    type: Literal[ModelProviderTypeEnum.MOONSHOT]
    moonshotFileId: Optional[str] = None
    _public_endpoint: str = "api.moonshot.cn"
    _model_uri = V1_MODELS_URI


class OllamaConfig(BaseProviderConfig):
    type: Literal[ModelProviderTypeEnum.OLLAMA]
    ollamaServerHost: Optional[str] = PydanticField(
        default=None, json_schema_extra={"field_required": True}
    )
    ollamaServerPort: Optional[int] = PydanticField(
        default=None, json_schema_extra={"field_required": True}
    )
    _default_schema = "http"
    _model_uri = V1_MODELS_URI

    def get_base_url(self):
        if not self.ollamaServerHost:
            return None
        port_suffix = f":{self.ollamaServerPort}" if self.ollamaServerPort else ""
        domain = f"{self.ollamaServerHost}{port_suffix}"
        return f"{self._default_schema}://{domain}"


class OpenAIConfig(BaseProviderConfig):
    type: Literal[ModelProviderTypeEnum.OPENAI]
    openaiCustomUrl: Optional[str] = None
    responseJsonSchema: Optional[dict] = None
    _public_endpoint: str = "api.openai.com"
    _model_uri = V1_MODELS_URI

    def get_base_url(self) -> Optional[str]:
        if self.openaiCustomUrl:
            parsed_url = urlparse(self.openaiCustomUrl)
            return f"{parsed_url.scheme}://{parsed_url.netloc}"
        return super().get_base_url()

    def get_model_url(self) -> Tuple[Optional[str], Optional[str]]:
        if not self.openaiCustomUrl:
            return super().get_model_url()
        parsed_url = urlparse(self.openaiCustomUrl)
        model_uri = f"{parsed_url.path.rstrip('/')}/models"
        return self.get_base_url(), model_uri

    def get_chat_url(self):
        if not self.openaiCustomUrl:
            return super().get_chat_url()
        parsed_url = urlparse(self.openaiCustomUrl)
        chat_uri = f"{parsed_url.path.rstrip('/')}/chat/completions"
        return self.get_base_url(), chat_uri


class OpenrouterConfig(BaseProviderConfig):
    type: Literal[ModelProviderTypeEnum.OPENROUTER]
    _public_endpoint: str = "openrouter.ai"


class QwenConfig(BaseProviderConfig):
    type: Literal[ModelProviderTypeEnum.QWEN]
    qwenEnableSearch: Optional[bool] = None
    qwenFileIds: Optional[List[str]] = None
    qwenEnableCompatible: Optional[bool] = None
    _public_endpoint: str = "dashscope.aliyuncs.com"
    _model_uri = "/compatible-mode/v1/models"
    _chat_uri = "/compatible-mode/v1/chat/completions"
    _default_override = {"qwenEnableCompatible": True}


class SparkConfig(BaseProviderConfig):
    type: Literal[ModelProviderTypeEnum.SPARK]
    _public_endpoint: str = "spark-api-open.xf-yun.com"


class StepfunConfig(BaseProviderConfig):
    type: Literal[ModelProviderTypeEnum.STEPFUN]
    _public_endpoint: str = "api.stepfun.com"


class TogetherAIConfig(BaseProviderConfig):
    type: Literal[ModelProviderTypeEnum.TOGETHERAI]
    _public_endpoint: str = "api.together.xyz"


class TritonConfig(BaseProviderConfig):
    type: Literal[ModelProviderTypeEnum.TRITON]
    modelVersion: Optional[str] = None
    tritonDomain: Optional[str] = None

    def get_base_url(self) -> Optional[str]:
        if not self.tritonDomain:
            return None
        return f"{self._default_schema}://{self.tritonDomain}"


class YiConfig(BaseProviderConfig):
    type: Literal[ModelProviderTypeEnum.YI]
    _public_endpoint: str = "api.lingyiwanwu.com"


class ZhipuaiConfig(BaseProviderConfig):
    type: Literal[ModelProviderTypeEnum.ZHIPUAI]
    _public_endpoint: str = "open.bigmodel.cn"


ProviderConfigType = Union[
    Ai360Config,
    AzureOpenAIConfig,
    BaichuanConfig,
    BaiduConfig,
    BedrockConfig,
    ClaudeConfig,
    CloudflareConfig,
    CohereConfig,
    CozeConfig,
    DeeplConfig,
    DeepseekConfig,
    DifyConfig,
    DoubaoConfig,
    FireworksConfig,
    GaladrielConfig,
    GeminiConfig,
    GithubConfig,
    GrokConfig,
    GroqConfig,
    HunyuanConfig,
    LongcatConfig,
    MinimaxConfig,
    MistralConfig,
    MoonshotConfig,
    OllamaConfig,
    OpenAIConfig,
    OpenrouterConfig,
    QwenConfig,
    SparkConfig,
    StepfunConfig,
    TogetherAIConfig,
    TritonConfig,
    YiConfig,
    ZhipuaiConfig,
]


class ProviderModel(BaseModel):
    name: str
    category: Optional[str] = None


class MaskedAPIToken(BaseModel):
    input: Optional[str] = None
    hash: Optional[str] = None

    @model_validator(mode="after")
    def check_fields(self):
        if self.input is None and self.hash is None:
            raise ValueError(
                "Either 'input' or 'hash' must be provided for a masked API token."
            )
        if self.input is not None and self.hash is not None:
            raise ValueError(
                "Only one of 'input' or 'hash' can be provided for a masked API token."
            )
        if self.input is not None and not self.input.strip():
            raise ValueError("API token input cannot be empty or just whitespace.")
        return self


class ModelProviderBase(SQLModel):
    name: str = Field(index=True, nullable=False)
    description: Optional[str] = Field(default=None, nullable=True)
    timeout: int = Field(default=120, nullable=False)
    config: ProviderConfigType = Field(
        description="provider specific configuration",
        sa_column=Column(
            pydantic_column_type(
                ProviderConfigType,
                exclude_defaults=True,
                exclude_none=True,
                exclude_unset=True,
            ),
        ),
    )
    models: Optional[List[ProviderModel]] = Field(
        default=[],
        sa_column=Column(
            pydantic_column_type(List[ProviderModel]),
            nullable=True,
        ),
    )
    proxy_url: Optional[str] = Field(default=None, nullable=True)
    proxy_timeout: Optional[int] = Field(default=None, nullable=True)

    @model_validator(mode="after")
    def check_all(self):
        if self.timeout <= 0:
            raise ValueError("timeout must be a positive integer")
        if self.proxy_timeout is not None and self.proxy_timeout <= 0:
            raise ValueError("proxy_timeout must be a positive integer")
        if self.proxy_timeout is not None and self.proxy_url is None:
            raise ValueError("proxy_url must be set when proxy_timeout is set")
        return self


class ModelProviderUpdate(ModelProviderBase):
    api_tokens: List[MaskedAPIToken] = PydanticField(
        default=[],
    )

    @field_validator("api_tokens")
    def check_api_tokens(cls, v):
        if v is not None:
            if not isinstance(v, list) or len(v) == 0:
                raise ValueError("api_tokens must be a non-empty list")
        return v


class ModelProviderCreate(ModelProviderUpdate):
    clone_from_id: Optional[int] = PydanticField(default=None)


class ModelProvider(ModelProviderBase, BaseModelMixin, table=True):
    __tablename__ = "model_providers"
    __table_args__ = (
        # Provider names are unique within their owning Org — two Orgs
        # can each have an "openai" provider without colliding.
        UniqueConstraint(
            'owner_principal_id', 'name', name='uix_model_providers_name_per_owner'
        ),
    )
    id: Optional[int] = Field(default=None, primary_key=True)
    # Every provider belongs to one Org. The route layer fills this
    # with ctx.current_principal_id, falling back to platform_principal_id
    # for admin in "All" mode (Global providers are not a thing — only
    # instance templates and inference backends carry a NULL-owner
    # Global notion).
    owner_principal_id: Optional[int] = Field(
        default=None,
        sa_column=Column(Integer, ForeignKey("principals.id"), nullable=False),
    )
    api_tokens: List[str] = Field(
        sa_column=Column(JSON, nullable=False),
        default=[],
    )
    model_route_targets: List["ModelRouteTarget"] = Relationship(
        back_populates="provider",
        sa_relationship_kwargs={"lazy": "noload", "cascade": "delete"},
    )

    @classmethod
    def _convert_to_public_class(cls, data) -> "ModelProviderPublic":
        # somehow when updating model provider while deleting targets
        # the result of await ModelProvider.one_by_id(session=session, id=id) is not fully correct.
        # e.g. the provider.config is a dict instead of correct config class and it will
        # yields validation warnings when model_dump it. So setting warnings=False to ignore
        # the warnings and convert it to correct config class by ourselves.
        dict_data = data if isinstance(data, dict) else data.model_dump(warnings=False)
        current_tokens: List[str] = dict_data.pop("api_tokens", None)
        masked_tokens: List[MaskedAPIToken] = []
        if current_tokens:
            masked_tokens = [
                {"hash": hashlib.sha256(token.encode()).hexdigest()}
                for token in current_tokens
            ]
        dict_data["api_tokens"] = masked_tokens
        return ModelProviderPublic.model_validate(dict_data)


class ModelProviderPublic(ModelProviderUpdate, PublicFields):
    # The owning Org. Server-set on create from the caller's tenant
    # context (the DB column is NOT NULL — providers have no "Global"
    # notion, unlike instance templates / inference backends). Kept
    # out of ModelProviderBase / Update so clients can't smuggle their
    # own tenant override; surfaced here for list / get responses.
    # Typed as a plain `int` so the generated OpenAPI / TS clients
    # treat it as required and non-nullable.
    owner_principal_id: int


ModelProvidersPublic = PaginatedList[ModelProviderPublic]


class ModelProviderListParams(ListParams):
    sortable_fields: ClassVar[List[str]] = [
        "id",
        "name",
        "created_at",
        "updated_at",
    ]


class ProviderModelsInput(BaseModel):
    api_token: Optional[str] = None
    config: Optional[ProviderConfigType] = None
    proxy_url: Optional[str] = None


class TestProviderModelInput(ProviderModelsInput):
    model_name: str


class TestProviderModelResult(BaseModel):
    model_name: str
    accessible: bool
    error_message: Optional[str] = None
