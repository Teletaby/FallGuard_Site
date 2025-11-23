# Quick Start Guide for Render Deployment

## Step-by-Step Deployment

### 1. **Prepare Your Repository**

Make sure you have a GitHub repository with all files. Your repo should have:
- ✅ `main.py`
- ✅ `requirements.txt`
- ✅ `Procfile`
- ✅ `runtime.txt`
- ✅ `render.yaml`
- ✅ `.gitignore`
- ✅ `app/` folder with application code
- ✅ `models/` folder with your trained model
- ✅ `data/` folder with data files

### 2. **Push to GitHub**

```bash
cd FallGuard_test-main
git init
git add .
git commit -m "Deploy to Render"
git remote add origin https://github.com/YOUR_USERNAME/fallguard.git
git branch -M main
git push -u origin main
```

### 3. **Create Render Service**

1. Go to https://dashboard.render.com
2. Click **"New +"** → **"Web Service"**
3. Select **"Build and deploy from a Git repository"**
4. Connect GitHub and select your `fallguard` repository
5. Fill in the details:

| Field | Value |
|-------|-------|
| Name | `fallguard` |
| Region | `Oregon` |
| Branch | `main` |
| Build Command | `pip install -r requirements.txt` |
| Start Command | `gunicorn --timeout 120 --workers 1 main:app` |

6. **Optional**: Add Environment Variables
   - Click "Add Environment Variable"
   - Key: `TELEGRAM_BOT_TOKEN`
   - Value: `your-bot-token`
   - Repeat for `ADMIN_PASSWORD` if desired

7. Click **"Create Web Service"**

### 4. **Wait for Deployment**

- Render will build and deploy automatically
- Check the **Logs** tab for any issues
- Deployment takes 2-5 minutes
- You'll get a public URL like: `https://fallguard-xxxx.onrender.com`

### 5. **Access Your App**

Once deployed:
- Open your Render URL in a browser
- Dashboard should load
- Check `/api/debug/cameras` for system status

## Troubleshooting

### Build Fails
- Check **Logs** tab in Render dashboard
- Common issues:
  - Missing files
  - PyTorch installation too slow → increase timeout
  - Model file too large → use Git LFS

### App Starts but Gives Error
- Check **Logs** for error messages
- Common issues:
  - Model file path incorrect
  - Missing environment variables
  - Dependencies not installed

### Service Spins Down
- Free tier behavior after 15 min inactivity
- Just access the URL again to restart

## Cost

- **Free Tier**: $0/month (limited resources, spins down after 15 min inactivity)
- **Paid Plans**: Start at $7/month (always running)

## Key Files Explained

- **Procfile**: Tells Render how to run your app
- **runtime.txt**: Specifies Python version
- **render.yaml**: Advanced Render configuration (optional)
- **requirements.txt**: All Python dependencies
- **.env.example**: Template for environment variables

## Next Steps

After deployment:

1. Monitor application health in Render dashboard
2. Check logs regularly for errors
3. Set up custom domain (if paid plan)
4. Enable automatic deployments (default)
5. Monitor performance and adjust resources if needed

## Support

- **Render Docs**: https://render.com/docs
- **Render Support**: support@render.com
- **Check Logs**: https://dashboard.render.com → Select your service → Logs tab
