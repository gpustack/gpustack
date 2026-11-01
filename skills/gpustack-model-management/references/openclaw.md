# Using this skill with OpenClaw

OpenClaw loads skills from `SKILL.md` folders and talks to OpenAI-compatible model servers through custom
providers, so it can both manage GPUStack models (this skill) and use them as its own model.

## Install the skill

Copy the skill folder into your OpenClaw workspace:

```bash
mkdir -p ~/.openclaw/workspace/skills
cp -r skills/gpustack-model-management ~/.openclaw/workspace/skills/
openclaw skills list    # gpustack-model-management should appear
```

The skill needs `curl` on the host, and its `GPUSTACK_URL` and `GPUSTACK_API_KEY` environment variables.
Either export them before starting OpenClaw, or set them in the skill's entry in the OpenClaw config.
OpenClaw applies these to the host process for the agent run (not inside a sandbox), and only where the
variable is not already set:

```json5
{
  skills: {
    entries: {
      "gpustack-model-management": {
        enabled: true,
        env: {
          GPUSTACK_URL: "http://your-gpustack-server",
          GPUSTACK_API_KEY: "gpustack_...",
        },
      },
    },
  },
}
```

Use an API key with `management` scope here. Ask the agent things like "list the models on GPUStack",
"deploy Qwen/Qwen3-8B with vLLM on cluster 1" or "scale qwen3-8b to 2 replicas".

## Use a GPUStack model as OpenClaw's model

Register GPUStack as a custom provider with the `openai-completions` adapter. The model `id` is the
GPUStack route name (as listed by `GET /v1/models`), and `baseUrl` ends in `/v1`:

```json5
{
  agents: {
    defaults: {
      model: { primary: "gpustack/qwen3-8b" },
    },
  },
  models: {
    mode: "merge",
    providers: {
      gpustack: {
        baseUrl: "http://your-gpustack-server/v1",
        apiKey: "gpustack_...",
        api: "openai-completions",
        models: [
          {
            id: "qwen3-8b",
            name: "Qwen3 8B (GPUStack)",
            contextWindow: 32768,
            maxTokens: 8192,
          },
        ],
      },
    },
  },
}
```

- Use a key with `inference` scope for the provider, separate from the management key above.
- Set `contextWindow` and `maxTokens` to what the deployment actually serves (for vLLM, its
  `--max-model-len`).
- A route owned by a non-platform GPUStack organization is addressed as `<org>/<route>`; use that as the
  model `id`.
- If the GPUStack model is served by llama.cpp and you want OpenClaw to use tools with it, add
  `compat: { supportsTools: true, toolSchemaProfile: "llamacpp" }` to the model entry, as OpenClaw
  recommends for llama.cpp servers.

See OpenClaw's [Creating skills](https://docs.openclaw.ai/tools/creating-skills) and
[custom providers](https://docs.openclaw.ai/gateway/config-tools/custom-providers) documentation for the
full configuration reference.
