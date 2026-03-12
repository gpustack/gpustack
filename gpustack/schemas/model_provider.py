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
from sqlmodel import (
    Field,
    Column,
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
    _model_uri = "/v1/models"


class BaiduConfig(BaseProviderConfig):
    type: Literal[ModelProviderTypeEnum.BAIDU]
    _public_endpoint: str = "qianfan.baidubce.com"
    _model_uri = "/v1/models"


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
    type: Literal[ModelProviderTypeEnum.CLAUDE]
    claudeVersion: Optional[str] = None
    _public_endpoint: str = "api.anthropic.com"
    _model_uri = "/v1/models"
    _chat_uri = "/v1/messages"


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
    _model_uri = "/v1/models"


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
    _model_uri = "/v1/models"


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
    _model_uri = "/v1/models"


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
    _model_uri = "/v1/models"


class OllamaConfig(BaseProviderConfig):
    type: Literal[ModelProviderTypeEnum.OLLAMA]
    ollamaServerHost: Optional[str] = PydanticField(
        default=None, json_schema_extra={"field_required": True}
    )
    ollamaServerPort: Optional[int] = PydanticField(
        default=None, json_schema_extra={"field_required": True}
    )
    _default_schema = "http"
    _model_uri = "/v1/models"

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
    _model_uri = "/v1/models"

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
    name: str = Field(index=True, nullable=False, unique=True)
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
    id: Optional[int] = Field(default=None, primary_key=True)
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
    pass


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
