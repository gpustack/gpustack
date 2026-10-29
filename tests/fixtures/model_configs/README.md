# Model metadata fixtures

These are public `config.json` snapshots for offline configuration inspection tests.
Only JSON is used; repository Python and model weights are never loaded.
Sources are `https://huggingface.co/<repository>/resolve/<revision>/config.json`.

`scheduler_cases.json` contains synthetic minimal metadata for architecture
classification and diffusion detection. These cases test routing decisions,
not complete model dimensions or upstream checkpoint fidelity.

| File | Repository | Revision |
| --- | --- | --- |
| qwen2.json | Qwen/Qwen2-0.5B-Instruct | c540970f9e29518b1d8f06ab8b24cba66ad77b6d |
| qwen3.json | Qwen/Qwen3-8B | b968826d9c46dd6066d109eabc6255188de91218 |
| qwen3_moe.json | Qwen/Qwen3-30B-A3B | ad44e777bcd18fa416d9da3bd8f70d33ebb85d39 |
| qwen3_vl.json | Qwen/Qwen3-VL-8B-Instruct | 0c351dd01ed87e9c1b53cbc748cba10e6187ff3b |
| qwen3_5.json | Qwen/Qwen3.5-9B | c202236235762e1c871ad0ccb60c8ee5ba337b9a |
| llama.json | unsloth/Llama-3.3-70B-Instruct | 99cd0d2c829e92a67c844f9144c2509632e5c87f |
| deepseek_v3.json | deepseek-ai/DeepSeek-R1 | 56d4cbbb4d29f4355bab4b9a39ccb717a14ad5ad |
| deepseek_v32.json | deepseek-ai/DeepSeek-V3.2 | a7e62ac04ecb2c0a54d736dc46601c5606cf10a6 |
| chatglm.json | THUDM/glm-4-9b-chat-1m | f8579ac096986e44ba24a5060d2fb79bedb56ef3 |
| phi4mm.json | microsoft/Phi-4-multimodal-instruct | 93f923e1a7727d1c4f446756212d9d3e8fcc5d81 |
| internvl.json | OpenGVLab/InternVL3-8B | 853e3a797a661694b1b8ece0cb72dc2b23e3dac9 |
| gemma.json | unsloth/gemma-3-4b-it | bf46152c47f5dd20b896357cb51abc4c03b8ee8c |
| mistral.json | mistralai/Mistral-7B-Instruct-v0.3 | c170c708c41dac9275d15a8fff4eca08d52bab71 |
| bert.json | BAAI/bge-large-en-v1.5 | d4aa6901d3a41ba39fb536a557fa166f842b0e09 |
| qwen_asr.json | Qwen/Qwen3-ASR-0.6B | 5eb144179a02acc5e5ba31e748d22b0cf3e303b0 |
| ming_flash_omni.json | inclusionAI/Ming-flash-omni-Preview | 7d26150c7d80f7b89dc04dc2a5264279c6be836a |
