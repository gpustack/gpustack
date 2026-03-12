from typing import Optional, List, Mapping
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


class ActiveConfig(BaseModel):
    activeProviderId: Optional[str] = Field(default=None)


class FailoverConfig(EnableState):
    healthCheckModel: Optional[str] = None
    failureThreshold: int = Field(default=1)


class AIProxyDefaultConfig(CustomConfig):
    id: str
    apiTokens: List[str] = Field(
        default_factory=list,
    )
    failover: FailoverConfig = Field(default_factory=FailoverConfig)
    retryOnFailure: EnableState = Field(default_factory=EnableState)
    type: ModelProviderTypeEnum
