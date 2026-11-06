---
translation_status: translated
---

# Использование аудиомоделей { #using-audio-models }

GPUStack поддерживает работу как моделей преобразования речи в текст, так и преобразования текста в речь. Модели преобразования речи в текст преобразуют аудиовходы на различных языках в письменный текст, а модели преобразования текста в речь преобразуют письменный текст в естественную и выразительную речь.

В этом руководстве мы покажем вам, как развернуть и использовать модели преобразования речи в текст и преобразования текста в речь в GPUStack.

## Предварительные условия { #prerequisites }

Прежде чем начать, убедитесь, что у вас есть следующее:

- GPUStack установлен и работает. Если нет, обратитесь к [Краткому руководству](../quickstart.md).
- Доступ к Hugging Face или ModelScope для загрузки файлов модели.

## Запуск модели преобразования речи в текст { #running-speech-to-text-model }

### Шаг 1. Развертывание модели преобразования речи в текст { #step-1-deploy-speech-to-text-model }

Выполните следующие шаги, чтобы развернуть модель из Каталога моделей:

1. Перейдите на страницу `Model Catalog` в пользовательском интерфейсе GPUStack.
2. Выберите `Speech-to-Text` в фильтре категории, затем выберите модель `Whisper-Large-V3-Turbo`.
3. Оставьте все по умолчанию и нажмите кнопку `Save`, чтобы развернуть модель.

![Развернуть модель](../assets/using-models/using-audio-models/deploy-stt-model.png)

После развертывания вы можете отслеживать состояние развертывания модели на странице `Deployments`. После успешного развертывания щелкните значок многоточия на развертывании и выберите `Open in Playground`, чтобы начать использовать модель в Песочнице.

![Список моделей](../assets/using-models/using-audio-models/stt-model-list.png)

### Шаг 2. Взаимодействие с моделью преобразования речи в текст { #step-2-interact-with-speech-to-text-model }

В Песочнице `Speech to Text`,

1. Нажмите кнопку `Upload`, чтобы загрузить аудиофайл, или нажмите кнопку `Microphone`, чтобы записать звук.
2. Нажмите кнопку `Generate Text Content`, чтобы сгенерировать транскрипцию.

![Сгенерировано](../assets/using-models/using-audio-models/inference-stt-model.png)

### Шаг 3. Потоковая передача вывода через API { #step-3-streaming-output-via-api }

Вы также можете использовать API для получения транскрипций потоковой передачи. Вот пример использования Curl:

```bash
# Replace ${SERVER_URL} with your GPUStack server URL and ${YOUR_GPUSTACK_API_KEY} with your API key.
curl ${SERVER_URL}/v1/audio/transcriptions \
  -H "Content-Type: multipart/form-data" \
  -H "Authorization: Bearer ${YOUR_GPUSTACK_API_KEY}" \
  -F model="whisper-large-v3-turbo" \
  -F file="@/path/to/audio-file;type=audio/mpeg" \
  -F language="en" \
  -F stream="true"
```

Результаты потоковой транскрипции будут возвращаться по мере их появления.

!!! note

    Потоковая транскрипция поддерживается только в том случае, если модель преобразования речи в текст развернута на серверной части `vLLM`. Серверная часть `VoxBox` не поддерживает потоковую передачу.

## Запуск модели преобразования текста в речь { #running-text-to-speech-model }

### Шаг 1. Развертывание модели преобразования текста в речь { #step-1-deploy-text-to-speech-model }

Выполните следующие шаги, чтобы развернуть модель из Каталога моделей:

1. Перейдите на страницу `Model Catalog` в пользовательском интерфейсе GPUStack.
2. Выберите `Text-to-Speech` в фильтре категории, затем выберите модель `Qwen3-TTS-12Hz-1.7B-CustomVoice`.
3. Оставьте все по умолчанию и нажмите кнопку `Save`, чтобы развернуть модель.

![Развернуть модель](../assets/using-models/using-audio-models/deploy-tts-model.png)

После развертывания вы можете отслеживать состояние развертывания модели на странице `Deployments`. После успешного развертывания щелкните значок многоточия на развертывании и выберите `Open in Playground`, чтобы начать использовать модель в Песочнице.

![Список моделей](../assets/using-models/using-audio-models/tts-model-list.png)

### Шаг 2. Взаимодействие с моделью преобразования текста в речь { #step-2-interact-with-text-to-speech-model }

В Песочнице `Text to Speech`,

1. Выберите нужный голос из раскрывающегося списка `Voice`.
2. (Необязательно) Укажите `Instructions`, чтобы модель генерировала желаемый стиль речи.
3. Введите текст, который хотите преобразовать в речь.
4. Нажмите кнопку `Submit`, чтобы создать звук.

![Сгенерировано](../assets/using-models/using-audio-models/inference-tts-model.png)

### Шаг 3. Потоковая передача вывода через API { #step-3-streaming-output-via-api_1 }

Вы также можете использовать API для вывода потокового аудио. Вот пример использования Curl:

```bash
# Replace ${SERVER_URL} with your GPUStack server URL and ${YOUR_GPUSTACK_API_KEY} with your API key.
curl ${SERVER_URL}/v1/audio/speech \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${YOUR_GPUSTACK_API_KEY}" \
  -d '{
    "model": "qwen3-tts-12hz-1.7b-customvoice",
    "voice": "Vivian",
    "task_type": "CustomVoice",
    "language": "Auto",
    "input": "Good one. Okay, fine, I'm just gonna leave this sock monkey here. Goodbye.",
    "stream": true,
    "response_format": "pcm"
  }' --no-buffer | play -t raw -r 24000 -e signed -b 16 -c 1 -
```

Это позволит напрямую передавать аудиовыход и воспроизводить его с помощью команды `play`. Звук передается в формате PCM с частотой дискретизации 24 кГц, 16-битным знаковым кодированием и моноканалом.

!!! note

    Потоковая передача речи поддерживается только в том случае, если модель преобразования текста в речь развернута на серверной части `vLLM`. Серверная часть `VoxBox` не поддерживает потоковую передачу.

## Клонирование голоса с помощью Qwen3-TTS { #voice-cloning-using-qwen3-tts }

GPUStack также поддерживает клонирование голоса с помощью моделей преобразования текста в речь. Вот как его использовать:

### Шаг 1. Развертывание модели клонирования голоса { #step-1-deploy-voice-cloning-model }

1. Перейдите на страницу `Model Catalog` в пользовательском интерфейсе GPUStack.
2. Выберите `Text-to-Speech` в фильтре категории, затем выберите модель `Qwen3-TTS-12Hz-1.7B-Base`.
3. Оставьте все по умолчанию и нажмите кнопку `Save`, чтобы развернуть модель.

### Шаг 2. Используйте клонирование голоса в Песочнице. { #step-2-use-voice-cloning-in-playground }

После успешного развертывания щелкните значок многоточия развертывания и выберите `Open in Playground`, чтобы начать использовать модель в Песочнице. Затем выполните следующие действия:

1. В поле `Reference Audio` загрузите аудиофайл или введите URL-адрес аудио, чтобы предоставить эталонный голос для клонирования. Например, вы можете ввести URL-адрес `https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-0115/APS-en_33.wav`.
2. Отметьте опцию `Use Speaker Embedding Only (no ICL)`.
3. Введите текст, который нужно синтезировать, например: `Good one. Okay, fine, I'm just gonna leave this sock monkey here. Goodbye.`
4. Нажмите кнопку `Submit`, чтобы сгенерировать речь с клонированным голосом.

![Клонирование голоса](../assets/using-models/using-audio-models/voice-clone.png)
