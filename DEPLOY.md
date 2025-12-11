# Deployment Guide: Render via GitHub

Your project is now fully configured for automatic deployment!
Because I cannot access your personal GitHub or Render accounts, you need to perform the **one-time** setup steps below.
After this, any changes you save and push will **automatically** update your website.

## Step 1: Push to GitHub

1.  **Log in to [GitHub](https://github.com)** and create a **New Repository**.
    *   Repository Name: `pollution_free_routing` (or whatever you prefer)
    *   **Do not** initialize with README, .gitignore, or License (I have already done this).
2.  **Copy** the commands shown under "…or push an existing repository from the command line".
3.  **Run** those commands in your terminal here (VS Code terminal). They will look typically like this:

    ```bash
    git remote add origin https://github.com/YOUR_USERNAME/pollution_free_routing.git
    git push -u origin main
    ```

    *(Note: You might need to log in to GitHub in the terminal window if prompted)*

## Step 2: Deploy on Render

1.  **Log in to [Render.com](https://render.com)** (you can use your GitHub account).
2.  Click **"New +"** and select **"Web Service"**.
3.  Connect your GitHub account if asked, then select the `pollution_free_routing` repository you just created.
4.  **Configuration**:
    *   Render will read the `render.yaml` file I created and should **automatically fill in** the settings.
    *   **Name**: `pollution-free-routing`
    *   **Region**: Singapore (or nearest to you)
    *   **Branch**: `main`
    *   **Runtime**: `Python 3`
    *   **Build Command**: `pip install -r requirements.txt` (Auto-filled)
    *   **Start Command**: `gunicorn wsgi:app` (Auto-filled)
    *   **Plan**: Free
5.  Click **"Create Web Service"**.

## Done!
Render will start building your app. It may take a few minutes.
Once finished, you will get a URL (e.g., `https://pollution-free-routing.onrender.com`).
**From now on, the specific "Zero Intervention" workflow is:**
1.  You ask me to make changes.
2.  I make the changes and commit them.
3.  You run `git push`.
4.  Render deploys automatically.
