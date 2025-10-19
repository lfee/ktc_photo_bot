# KTC Photo Bot

A Telegram bot for editing photos using AI services (OpenAI and Google Vision). Built with Python, aiogram, OpenCV and Pillow.

## Contents

- `bot.py` — main bot entrypoint and FSM logic
- `example.py` — example usage / scripts
- `from google` package — Google integration helpers

## Features

- Receive photos from Telegram users and apply AI-powered edits
- Supports OpenAI Image Edit API and Google Vision-based processing
- Local preprocessing with OpenCV / Pillow and temporary file handling
- FSM-driven conversation flow (aiogram) with memory storage

## Requirements

- Python 3.10+ recommended
- OS: Windows / Linux / macOS (development target: Windows)

## Environment variables

Create a `.env` in the project root or set these in your environment:

- `TELEGRAM_TOKEN` — Telegram bot token
- `OPENAI_API_KEY` — OpenAI API key (for image edits)
- `GOOGLE_API_KEY` — Google Vision API key (if using Google integration)

## Install dependencies

Install the minimal dependencies used by this project. Adjust / pin versions as needed.

```powershell
python -m pip install --upgrade pip
pip install aiogram opencv-python pillow python-dotenv aiohttp
```

If you use a virtual environment on Windows (recommended):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt  # if you create one
```

## Configuration constants

These conventions are used by the bot (see `bot.py`):

- `MAX_IMAGE_SIZE = 4 * 1024 * 1024` — max incoming image size (4 MB)
- `MAX_RETRIES = 3` — retry attempts for external API calls
- `CONCURRENT_REQUESTS_LIMIT = 10` — semaphore limit for parallel requests

## How it works (high level)

1. User sends a photo to the bot.
2. Bot validates size and saves it as `tmp_{chat_id}_{message_id}.jpg` in a temporary directory.
3. Bot asks user to pick a service (OpenAI / Google Vision) and to provide an editing prompt.
4. Bot performs processing asynchronously (uses executor for long-running OpenAI calls).
5. Result saved as `{original_path}.edited.png` and sent back to the user.
6. Temporary files are cleaned up in finally blocks.

## FSM / Conversation flow

- Uses aiogram 3.x FSM (see `bot.py`)
- States: `choosing_service` → `waiting_for_prompt` → `waiting_for_result`

## Prompt guidance

Examples and recommendations for good prompts (adapted per service):

- Style transfer: "Transform this photo into an aquarelle painting."
- Color correction: "Make the photo more contrasty and saturated."
- Background change: "Replace the background with a sunset by the sea."
- Portrait touch-up: "Enhance the face to be sharper and reduce blemishes."

Tips:

- Keep prompts focused and concise. Specify one main effect.
- Account for API limits (image size, types of effects).

## Performance and production notes

- Compress large input images before sending to external APIs.
- Use an asyncio.Semaphore to limit concurrent external calls.
- Cache frequently used results where possible to reduce API calls.
- Set network timeouts: e.g., 180s for OpenAI image edits, 60s for image downloads.

## Error handling and retries

- Wrap external API calls with retry logic (up to `MAX_RETRIES`).
- Notify the user on API errors and offer to retry.
- Always remove temporary files in a finally block to avoid disk bloat.

## Development tips

- Validate environment variables before starting the bot.
- Add logging to track requests, latencies and errors.
- Consider adding rate-limiting per-user to prevent abuse.

## Troubleshooting

- Telegram bot doesn't start: ensure `TELEGRAM_TOKEN` is set and valid.
- Image processing fails: check `OPENAI_API_KEY` / `GOOGLE_API_KEY` and quotas.
- Too many concurrent requests: reduce `CONCURRENT_REQUESTS_LIMIT`.

## Next improvements (suggested)

- Add `requirements.txt` with pinned versions and CI checks.
- Add unit tests for prompt handling and temp-file lifecycle.
- Add Dockerfile for deploying the bot in a containerized environment.

## License

This repository does not include an explicit license file. Add one if you plan to publish the project.

---

If you want, I can also:

- create a `requirements.txt` with pinned dependencies,
- add a minimal `Procfile` / `Dockerfile`, or
- add a quick `run_bot.ps1` for Windows to start the bot.
