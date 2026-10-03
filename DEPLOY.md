# Put the project online as a website

These steps give you a public link such as `https://your-name-diabetes-prediction.hf.space` that anyone can open in a browser. Nothing needs to be installed on the visitor's computer.

The recommended host is **Hugging Face Spaces**:
- it is free, and no credit card is needed;
- it provides 16 GB of RAM;
- it builds and runs the `Dockerfile` in this project automatically.

## What happens when it builds

The `Dockerfile` does the following:

1. Builds the React app.
2. Installs the Python libraries.
3. Trains the model inside the container, so the saved model always matches the installed library versions.
4. Runs the backend test suite. If anything fails, the build stops instead of publishing a broken site.
5. Starts one server on port 7860:
   - website pages at `/`, `/predict`, `/results`, and so on;
   - API at `/api/...`;
   - API documentation at `/docs`.

The first build takes about **8–12 minutes**. After that, the site starts in seconds.

---

## Step-by-step: Hugging Face Spaces

### 1. Create an account
Sign up at https://huggingface.co/join and verify your email address.

### 2. Create a Space
1. Go to https://huggingface.co/new-space.
2. **Space name:** for example `diabetes-prediction`.
3. **License:** any; MIT is fine.
4. **SDK:** choose **Docker**, then the **Blank** template.
5. **Hardware:** the free CPU option.
6. **Visibility:** Public, so examiners can open it.
7. Click **Create Space**.

### 3. Upload the project

**Option A: in the browser (easiest)**
1. Unzip `diabetes-prediction-system.zip` on your computer.
2. On your Space page, open **Files**, then **Add file**, then **Upload files**.
3. Open the unzipped `diabetes-prediction-system` folder. Select **everything inside it** and drag it into the upload area. Do not drag the outer folder itself. The items to include are:
   - `Dockerfile`
   - `README.md`
   - `.dockerignore`
   - `backend/`
   - `frontend/`
   - `outputs/`
   - `docs/`
   - `e2e/`
   - the other top-level files
4. Click **Commit changes to main**.

> The `README.md` must be the one from this project. Its first lines (between the `---` markers) tell Hugging Face to use Docker on port 7860.
>
> Do not upload `node_modules/` or `.venv/` if you created them locally.

**Option B: with git**

```bash
git clone https://huggingface.co/spaces/YOUR-USERNAME/diabetes-prediction
# copy everything from diabetes-prediction-system/ into the cloned folder, then:
cd diabetes-prediction
git add .
git commit -m "Diabetes Prediction System"
git push
```

When git asks for a password, use a Hugging Face **access token** with *write* permission. Create one at https://huggingface.co/settings/tokens.

### 4. Wait for the build
- The Space shows **Building**. Click **Logs** to watch progress; you will see the `[train]` lines from model training.
- When the status changes to **Running**, the site is live.
- Your link is `https://YOUR-USERNAME-diabetes-prediction.hf.space`. It is also embedded on the Space page.

### 5. Check it
1. Open the link. The Dashboard should show **Connected** and **Support Vector Machine (RBF)**.
2. Go to **Prediction**, then click **Load example**, then **Predict**.

---

## Good to know

- **Sleeping.** Free Spaces go to sleep after a period with no visitors. The next visit wakes the site, which takes about a minute. Open it shortly before your viva.
- **Updating.** Upload the changed files again (or `git push`). The Space rebuilds automatically.
- **Same results as on your computer.** Training uses a fixed `random_state`, so the metrics match your local run.
- **Disclaimer.** The medical disclaimer is shown on every page of the public site.

## If the build fails
- Open **Logs**. The last lines show which step failed.
- **"Space configuration" or port error.** Check that `README.md` starts with the `---` block containing `sdk: docker` and `app_port: 7860`.
- **Test failure during the build.** The log shows which test failed. Fix it and push again.

## Alternative: Render.com
The same `Dockerfile` also works on Render:
1. Push the project to GitHub.
2. In Render, create a **New Web Service** from that repository and choose the **Docker** runtime.

Render sets `$PORT` automatically. The running server uses about 460 MB of memory, which is very close to the limit of Render's free 512 MB instance, so a paid instance with at least 1 GB is advisable there. This is why Hugging Face Spaces is the recommended option.
