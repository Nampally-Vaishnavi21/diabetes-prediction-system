# Put the project online for free (Render)

At the end you get a public link like `https://diabetes-prediction.onrender.com`. Anyone can open it in a browser: the website, the API and the trained model all run in one container.

**Why Render.** It runs Docker containers on its free plan with 512 MB of memory, and no credit card is needed to start. This app uses about 210 MB of memory while running.

**What you need:**
- a GitHub account (free);
- a Render account (free; you can sign in with GitHub);
- `git` installed on your computer (check with `git --version`).

---

## Part 1: Put the project on GitHub

### 1. Create an empty repository
1. Go to **https://github.com/new**.
2. **Repository name:** `diabetes-prediction-system`.
3. **Public** or Private: either works with Render.
4. Do **not** tick "Add a README", ".gitignore" or "license". The repository must be empty.
5. Click **Create repository**.

### 2. Push the project
Open a terminal **inside the unzipped `diabetes-prediction-system` folder**. That is the folder containing `Dockerfile`, `backend/` and `frontend/`. Then run:

```bash
git init
git add .
git commit -m "Diabetes Prediction System"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/diabetes-prediction-system.git
git push -u origin main
```

- If git asks you to sign in, use your browser sign-in. If it asks for a password, use a GitHub **personal access token** instead of your password.
- `.gitignore` already leaves out `node_modules`, `.venv`, `frontend/dist` and the locally trained model files. They are not needed, because the server trains its own model.

Refresh the GitHub page. You should see `Dockerfile`, `backend/`, `frontend/` and the other files at the top level.

---

## Part 2: Deploy on Render

### 3. Create the account
Go to **https://render.com**, click **Get Started**, then **Sign up with GitHub**.

### 4. Create the web service
1. In the Render Dashboard, click **+ New**, then **Web Service**.
2. Choose **Git Provider** → **GitHub**. Allow Render to access your `diabetes-prediction-system` repository, then select it.
3. Fill in the settings:

| Setting | Value |
|---|---|
| Name | `diabetes-prediction` (this becomes your link) |
| Language | **Docker**. You must pick this yourself from the list. |
| Branch | `main` |
| Region | **Singapore** (closest to India) |
| Root Directory | leave empty (the `Dockerfile` is at the top level) |
| Instance Type | **Free** |

4. Open **Advanced** and set **Health Check Path** to `/api/health`. No environment variables are needed, because Render provides `PORT` itself.
5. Click **Create Web Service**.

### 5. Wait for the first build (about 10–20 minutes)
The **Logs** tab shows each stage:
- `npm ci` / `npm run build`: the React website builds;
- `pip install`: the Python libraries install;
- `[train] ...`: the model trains (single-threaded to stay within the free plan's limits);
- `76 passed`: the tests pass. If any test fails, the build stops and the old version stays online.
- `Uvicorn running on http://0.0.0.0:10000`: the server has started.

When the status at the top shows **Live**, your site is online.

### 6. Check it
1. Open `https://diabetes-prediction.onrender.com`. The exact link is shown at the top of the service page.
2. The **Dashboard** should show **Connected** and **Support Vector Machine (RBF)**.
3. Go to **Prediction**, click **Load example**, then **Predict**. The result page should appear.

Put this link in your report.

---

## Good to know

- **Sleeping.** On the free plan, the service goes to sleep after 15 minutes with no visitors. The next visit wakes it in about a minute; after that it is fast. **Open the link a few minutes before your viva.**
- **Free hours.** Render gives 750 free instance hours per month, which is enough for this one service to stay available all month.
- **Updating.** Change files locally, then run `git add .`, `git commit -m "update"` and `git push`. Render rebuilds and redeploys automatically.
- **Same results.** Training uses a fixed `random_state`, so the online model gives the same metrics as your local run.

## If something goes wrong

| What you see | Fix |
|---|---|
| Build fails at `npm ci` | Check that `frontend/package-lock.json` was pushed to GitHub. |
| Build fails with a test failure | The log names the failing test. Paste the last 30 lines of the log to get help. |
| Build fails with "out of memory" or the build is killed | Check that the `Dockerfile` contains `TRAIN_N_JOBS=1`. |
| The site shows only `{"message": ...}` JSON | The React build was not copied. Check the log for errors after `npm run build`. |
| "Backend unavailable" right after waking | Wait 30 seconds and refresh. The server is still loading the model. |
| Deploy status "Failed" with a health check error | Check that **Health Check Path** is `/api/health`. |
