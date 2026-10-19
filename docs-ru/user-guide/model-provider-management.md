---
translation_status: translated
---

# Управление поставщиком моделей { #model-provider-management }

GPUStack использует возможности Higress для обеспечения публичной интеграции MaaS. На странице `Model` - `Provider` вы можете управлять интеграцией сервисов общедоступных моделей.

В настоящее время поддерживаются следующие поставщики:

- AI360
- Microsoft Azure
- Baichuan Intelligent Technology
- Baidu AI Cloud
- Amazon Bedrock
- Anthropic Claude
- Cloudflare AI
- Cohere
- Coze
- DeepL
- DeepSeek
- Dify
- Doubao
- Fireworks AI
- Galadriel
- Google Gemini
- Generic Provider
- GitHub Copilot
- xAI Grok
- Groq
- Tencent Hunyuan
- LongCat AI
- MiniMax
- Mistral AI
- Moonshot AI
- Ollama
- OpenAI
- OpenRouter
- Alibaba Qwen
- iFLYTEK SparkDesk
- StepFun
- Together AI
- NVIDIA Triton Inference Server
- 01 AI
- Zhipu AI

## Создать поставщика { #create-provider }

1. Перейти на страницу `Providers`.
2. Нажмите `Add Provider`.
3. Заполните необходимые параметры, например `Name`, `Type`, `API Key`.
4. Нажмите `Add Model`, чтобы настроить хотя бы одну модель для этого поставщика.
5. Нажмите кнопку `Save`.

## Использование конечной точки, совместимой с Anthropic { #use-an-anthropic-compatible-endpoint }

Провайдер `Anthropic Claude` по умолчанию обращается к `api.anthropic.com`. Чтобы использовать собственный или проксируемый сервис с тем же API, задайте в `claudeCustomUrl` абсолютный URL `http(s)` с портом и префиксом пути, например `http://192.168.50.14:8080` или `https://gateway.example.com/anthropic`.

Указывайте адрес сервиса и префикс, под которым он размещён, **а не** полный базовый путь API: пути Anthropic добавляются автоматически. Поэтому `claudeCustomUrl` не должен оканчиваться на `/v1`, хотя в документации большинства совместимых сервисов этот суффикс входит в базовый URL. При `http://192.168.50.14:8080/v1` GPUStack обратится к `/v1/v1/models` и `/v1/v1/messages`: обычно `Test Model` завершается ошибкой, хотя сам инференс достигает сервиса. У `openaiCustomUrl` правило обратное: это полный базовый путь API с `/v1`. Учётные данные, строка запроса и фрагмент в URL не допускаются; API-ключ указывается в отдельном поле провайдера.

По умолчанию такой провайдер всё равно доступен через API OpenAI: на пути к сервису запросы преобразуются в протокол Anthropic, а ответы преобразуются обратно. Чтобы передавать их без изменений, добавьте `protocol: original` в конфигурацию провайдера. Сквозная передача сохраняет возможности, которых нет в схеме OpenAI: кэширование промптов, блоки рассуждений и вызовов инструментов. Однако провайдер в этом случае обслуживает **только протокол Anthropic**: запросы OpenAI к `/v1/chat/completions` больше не преобразуются. Используйте пути Anthropic (`/v1/messages`).

## Добавить маршрут для провайдера { #add-route-for-provider }

1. Перейти на страницу `Providers`.
2. Найдите Провайдера, для которого вы хотите создать маршрут.
3. Нажмите `Add Route` в `Operations` этого провайдера.
4. При необходимости измените маршрут `Name` и `Route Targets` для провайдера.
5. Нажмите кнопку `Save`.

## Редактировать поставщика { #edit-provider }

1. Перейти на страницу `Providers`.
2. Найдите поставщика, которого хотите изменить.
3. При необходимости измените имя, тип и атрибуты ключа API. При необходимости добавляйте/удаляйте модели из списка моделей.
4. Нажмите кнопку `Save`.

## Удалить поставщика { #delete-provider }

1. Перейти на страницу `Providers`.
2. Найдите поставщика, которого хотите удалить.
3. Нажмите кнопку `Delete` в столбце `Operations`.
4. Подтвердите удаление.
