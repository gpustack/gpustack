from typing import Dict, Optional, List, Mapping
from pydantic import BaseModel, Field
from gpustack.schemas.model_provider import ModelProviderTypeEnum


class EnableState(BaseModel):
    enabled: bool = Field(default=False)


class CustomConfig(BaseModel):
    azureServiceUrl: Optional[str] = None
    awsAccessKey: Optional[str] = None
    awsSecretKey: Optional[str] = None
    awsRegion: Optional[str] = None
    bedrockAdditionalFields: Optional[dict] = None
    claudeVersion: Optional[str] = None
    cloudflareAccountId: Optional[str] = None
    targetLang: Optional[str] = None
    difyApiUrl: Optional[str] = None
    botType: Optional[str] = None
    inputVariable: Optional[str] = None
    outputVariable: Optional[str] = None
    doubaoDomain: Optional[str] = None
    geminiSafetySetting: Optional[Mapping[str, str]] = None
    apiVersion: Optional[str] = None
    geminiThinkingBudget: Optional[float] = None
    hunyuanAuthId: Optional[str] = None
    hunyuanAuthKey: Optional[str] = None
    minimaxApiType: Optional[str] = None
    minimaxGroupId: Optional[str] = None
    moonshotFileId: Optional[str] = None
    ollamaServerHost: Optional[str] = None
    ollamaServerPort: Optional[int] = None
    openaiCustomUrl: Optional[str] = None
    responseJsonSchema: Optional[dict] = None
    qwenEnableSearch: Optional[bool] = None
    qwenFileIds: Optional[List[str]] = None
    qwenEnableCompatible: Optional[bool] = None
    modelVersion: Optional[str] = None
    tritonDomain: Optional[str] = None
    # Generic in the plugin, not per-provider: providerDomain overrides whatever
    # authority the provider hardcoded, providerBasePath is prepended to the
    # rewritten path, and protocol chooses between exposing OpenAI ("openai",
    # the plugin's default) and passing the provider's own protocol through
    # ("original"). ClaudeConfig.claudeCustomUrl is expressed with the first
    # two; protocol is left to whoever writes the provider config.
    protocol: Optional[str] = None
    providerDomain: Optional[str] = None
    providerBasePath: Optional[str] = None
    # Per-API path overrides, keyed by the plugin's ``ApiName`` strings. Entries
    # merge *over* the provider type's own defaults (the plugin only fills in
    # keys the config left out), so declaring one API keeps all the others.
    #
    # Declaring an API is also what makes ai-proxy treat it as natively served:
    # an inbound ``/v1/messages`` is rewritten to ``/v1/chat/completions``
    # exactly when the active provider has no ``anthropic/v1/messages``
    # capability. Note the plugin filters this map against a whitelist and drops
    # unknown keys *silently*, so a value that never takes effect looks
    # identical to one that was never set.
    capabilities: Optional[Dict[str, str]] = None


class ActiveConfig(BaseModel):
    activeProviderId: Optional[str] = Field(default=None)


class FailoverConfig(EnableState):
    healthCheckModel: Optional[str] = None
    failureThreshold: int = Field(default=1)


class AIProxyDefaultConfig(CustomConfig):
    id: str
    # Optional (rather than an empty-list default) so that ``exclude_none`` is
    # enough to omit it: a provider with no static credential must not emit
    # ``apiTokens: []``, or ai-proxy stops falling back to the inbound
    # ``Authorization`` header.
    apiTokens: Optional[List[str]] = None
    failover: FailoverConfig = Field(default_factory=FailoverConfig)
    retryOnFailure: EnableState = Field(default_factory=EnableState)
    type: ModelProviderTypeEnum
