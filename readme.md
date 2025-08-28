<p align="center"><img width="895" height="1024" alt="image" src="https://github.com/user-attachments/assets/dc7c59f4-8d1b-432b-8a01-4aa14b84b53d" /></p>

# <img src="icon.png" alt="Cloud Interrogator Logo" width="32"> Cloud Interrogator

Hiii!~ I made Cloud Interrogator because I wanted a cozy, beautiful little spot on my desktop to interact with machine vision models. It's a native app built with PySide6 that feels right at home on your computer.

Best of all, it's designed to feel like it was never closed. The app saves its entire state when you exit - your prompts, the image you were working with, your model settings, and even the window position—so you can always pick up exactly where you left off!

## Features

- **A UI That Feels Like Home**: A clean, modern interface that automatically uses your system's accent color (on Windows) for that truly native feel.
- **Connect to Anything**: The Endpoint Manager lets you connect to any OpenAI-compatible service. Add, edit, and save all your different APIs and models in one place.
- **Watch the Magic Happen**: See the model's response generate token-by-token in real time.
- **Never Lose Your Place**: Full persistence means you can close the app and come back later without losing your context. It's like a conversation that never ends!
- **Total Prompt Control**: Load local images, switch between separate System and User prompts, and even save/load your favorite image-and-prompt pairs to a file.
- **Fine-Tuning Knobs**: All the controls you need to get the perfect response: max tokens, temperature, top-p, and a seed for reproducible results.

## How to Use

1.  **Set Up Your Endpoint**: The first time you run the app, click **Manage Endpoints** to add your API details (give it a name, add the URL, your key, and a list of models).
2.  **Pick Your Model**: Choose your endpoint from the main dropdown menu.
3.  **Load an Image**: Click **Load Image** to pick a file.
4.  **Write Your Prompts**: Fill in the User and System prompt boxes.
5.  **Run!**: Adjust your settings and click **Run** to start the magic!

## Installation

### Portable Windows Binary

If messing around with Python isn't your jam, I've got you covered. You can grab the latest build directly from the [**Releases page**](https://github.com/Artificial-Sweetener/CloudInterrogator/releases)!

### With Python

There are two ways to run the app with Python.

#### The Easy Way (Windows)

Just double-click `run.bat`.

The first time you run it, a script will automatically create a virtual environment (`.venv`), install all the required packages, and add `winaccent` for native theme support. Every time you run it after that, it will just launch the app. Super simple!

For more control, you can use `venv.bat` to open a command prompt with the environment already activated.

#### Manual Setup (All Platforms)

If you prefer to manage the environment yourself, you can follow these steps. This method SHOULD works on Windows, macOS, and Linux, but please know that I haven't had a chance to test it on a Linux or macOS machine myself!

1.  **Create and activate a virtual environment**:
    ```bash
    # On Windows
    python -m venv .venv
    .\venv\Scripts\activate

    # On macOS & Linux
    python3 -m venv .venv
    source .venv/bin/activate
    ```
2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
3.  **(Optional) For native theme on Windows**:
    ```bash
    pip install winaccent
    ```
4.  **Run the app**:
    ```bash
    # On Windows
    python main.py

    # On macOS & Linux
    python3 main.py
    ```

## Building from Source

Want to compile the app yourself? This project uses Nuitka to create a single, neat executable.

1.  **Install Dependencies**: Make sure you've installed everything from `requirements.txt`.
2.  **Install Nuitka**: `pip install nuitka`
3.  **Run the Build Script**: Just run the `build.bat` file!

    ```bash
    .\build.bat
    ```

The provided `build.bat` is for Windows. It requires a C++ compiler, but don't worry if you don't have one! Nuitka is smart enough to offer to download what it needs.

The final `CloudInterrogator.exe` file will show up in the `build/` directory when it's done.

## From the Developer 💖

I hope you love using Cloud Interrogator as much as I loved building it! If you'd like to support my work or see what else I'm up to, here are a few links:

- **Recommended Inference Provider**: Looking for a great, fast AI provider to use alongside my app? I love using [Featherless.ai](https://featherless.ai/)! They charge a reasonable fee per month instead of per-token which I prefer because it makes my costs predictable, and they're one of the only inference providers I know of that is dedicated to privacy; no logs.
- **My Referral Code**: If you sign up with [this link](https://featherless.ai/register?referrer=4Z0BCRPO), it helps support me directly. Thank you! 🥰
- **Buy Me a Coffee**: You can help fuel more projects like this at my [Ko-fi page](https://ko-fi.com/artificial_sweetener).
- **My Website & Socials**: See my art, poetry, and other dev updates at [artificialsweetener.ai](https://artificialsweetener.ai).
- **If you like this project**, it would mean a lot to me if you gave me a star here on Github!! ⭐
