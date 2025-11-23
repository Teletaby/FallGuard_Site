# FallGuard - AI Fall Detection System

A Flask-based web application that uses AI (LSTM models) and computer vision to detect falls in real-time.

## Features

- Real-time fall detection using LSTM neural networks
- MediaPipe pose estimation for skeleton tracking
- Telegram notifications for fall alerts
- Web-based dashboard for monitoring
- Multi-camera support
- Admin panel for system configuration

## Deployment to Render

### Prerequisites

1. **GitHub Account** - Push your code to GitHub
2. **Render Account** - Sign up at [render.com](https://render.com)
3. **Environment Variables** (optional but recommended):
   - `TELEGRAM_BOT_TOKEN` - Your Telegram bot token
   - `ADMIN_PASSWORD` - Custom admin password
   - `PORT` - Server port (Render will set this automatically)

### Deployment Steps

#### 1. Push Code to GitHub

```bash
# Initialize git (if not already done)
git init

# Add files
git add .

# Commit
git commit -m "Initial commit for Render deployment"

# Add remote and push (replace with your GitHub repo URL)
git remote add origin https://github.com/YOUR_USERNAME/fallguard.git
git branch -M main
git push -u origin main
```

#### 2. Create a New Web Service on Render

1. Go to [render.com](https://render.com)
2. Click **"New +"** → **"Web Service"**
3. Select **"Deploy an existing repository"**
4. Connect your GitHub account if not already connected
5. Select the `fallguard` repository
6. Configure the following:

| Setting | Value |
|---------|-------|
| **Name** | fallguard (or your preferred name) |
| **Environment** | Python 3 |
| **Region** | Oregon (or your preferred region) |
| **Branch** | main |
| **Build Command** | `pip install -r requirements.txt` |
| **Start Command** | `gunicorn --timeout 120 --workers 1 main:app` |
| **Plan** | Free (or paid if preferred) |

#### 3. Add Environment Variables (Optional)

In the **Environment** section:

```
TELEGRAM_BOT_TOKEN=your_telegram_bot_token_here
ADMIN_PASSWORD=your_custom_admin_password
```

#### 4. Deploy

Click **"Create Web Service"** and Render will:
- Build your application
- Install dependencies from `requirements.txt`
- Start the server using `gunicorn`
- Provide you with a public URL

### Important Notes

#### Free Tier Limitations

- Services spin down after 15 minutes of inactivity
- No persistent storage (files will be lost)
- Limited resources

#### Model File

The pre-trained LSTM model file (`models/skeleton_lstm_pytorch_model.pth`) should be committed to your repository. If it's too large (>100MB), consider:

1. Using Git LFS (Large File Storage)
2. Downloading it from a remote source on startup
3. Training a smaller model

#### Uploads Directory

The `uploads/` directory won't persist. For production, consider:

- Using a cloud storage service (AWS S3, Google Cloud Storage, etc.)
- Using Render's Disk service for persistent storage

#### Camera Input

On Render, direct camera/webcam access won't work. The system will fall back to other input methods if configured.

### Accessing Your Deployment

Your application will be accessible at:
```
https://fallguard-[your-instance-id].onrender.com
```

- **Dashboard**: `https://your-url/`
- **Admin Panel**: Password required
- **Debug Info**: `https://your-url/api/debug/cameras`

### Troubleshooting

#### Build Fails
- Check the build logs in Render dashboard
- Ensure all requirements are in `requirements.txt`
- Verify Python version compatibility

#### Model Loading Error
- Model file must be in the repository
- Check file path is correct: `models/skeleton_lstm_pytorch_model.pth`

#### Service Spins Down
- Normal behavior on free tier after 15 min of inactivity
- Manual requests will wake it up

#### Port Issues
- Render automatically assigns the port via `PORT` environment variable
- Gunicorn command uses `--port $PORT` via the start command

## Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
python main.py
```

Access at `http://localhost:5000`

## Requirements

- Python 3.10+
- PyTorch
- OpenCV
- MediaPipe
- Flask
- Gunicorn

See `requirements.txt` for all dependencies.

## License

[Your License Here]

## Support

For issues or questions, please create an issue in the GitHub repository.
