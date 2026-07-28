# README

## Prerequisites

* Python 3.9 or newer (recommended)
* An OpenAI API key

## Setup

1. Clone this repository:

   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Copy the example environment file:

   ```bash
   cp .env.example .env
   ```

   > **Windows (Command Prompt):**
   >
   > ```cmd
   > copy .env.example .env
   > ```

3. Open `.env` and replace the placeholder value with your OpenAI API key:

   ```env
   API_SECRET_KEY="your-openai-api-key"
   ```

4. Run `pip install -r requirements.txt`

## Run the Script

Run the application with:

```bash
python src/main.py
```

If your system uses `python3` instead of `python`, run:

```bash
python3 src/main.py
```

## Notes

* Keep your `.env` file private and never commit it to version control.
* The `.env.example` file is only a template and should not contain real API keys.
