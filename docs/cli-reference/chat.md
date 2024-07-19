# gpustack chat

Chat with a large language model.

```bash
gpustack chat model [prompt]
```

## Positional arguments

| Name   | Description                                 |
| ------ | ------------------------------------------- |
| model  | The model to use for chat.                  |
| prompt | The prompt to send to the model. [Optional] |

## One-time chat with a prompt

If a prompt is provided, it performs a one-time inference. For example:

```bash
gpustack chat llama3 "tell me a joke."
```

Example output:

```
Why couldn't the bicycle stand up by itself?

Because it was two-tired!
```

## Interactive chat

If the `prompt` argument is not provided, you can chat with the large language model interactively. For example:

```bash
gpustack chat llama3
```

Example output:

```
>tell me a joke.
Here's one:

Why couldn't the bicycle stand up by itself?

(wait for it...)

Because it was two-tired!

Hope that made you smile!
>Do you have a better one?
Here's another one:

Why did the scarecrow win an award?

(think about it for a sec...)

Because he was outstanding in his field!

Hope that one stuck with you!

Do you want to hear another one?
>\quit
```

### Interactive commands

Followings are available commands in interactive chat:

```
Commands:
  \q or \quit - Quit the chat
  \c or \clear - Clear chat context in prompt
  \? or \h or \help - Print this help message
```

## Connect to external GPUStack server

If you are not running `gpustack chat` on the server node, you should provide the following environment variables:

| Name                | Description                                          |
| ------------------- | ---------------------------------------------------- |
| GPUSTACK_SERVER_URL | URL of the GPUStack server. e.g., `http://myserver`. |
| GPUSTACK_API_KEY    | GPUStack API key.                                    |
