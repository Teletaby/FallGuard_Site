# 📋 Render Deployment Checklist

Use this checklist to ensure everything is ready for deployment.

## ✅ Pre-Deployment Checklist

### Code Repository
- [ ] Project has a GitHub repository
- [ ] All code is committed to Git
- [ ] Repository is public or connected to Render
- [ ] Main branch is up to date

### Project Files
- [ ] `main.py` exists and is updated for environment variables
- [ ] `requirements.txt` has all dependencies
- [ ] `Procfile` exists with correct start command
- [ ] `runtime.txt` exists with Python version
- [ ] `render.yaml` exists (optional but helpful)
- [ ] `.gitignore` exists
- [ ] `models/skeleton_lstm_pytorch_model.pth` exists and is committed
- [ ] `app/` folder is complete with all modules
- [ ] `data/` folder has necessary data files

### Environment Configuration
- [ ] `.env.example` file created and documented
- [ ] Understand which environment variables are needed
- [ ] Have Telegram bot token ready (if using Telegram)
- [ ] Know your desired admin password

### Local Testing
- [ ] Run `setup_local.bat` (Windows) or `setup_local.sh` (Mac/Linux)
- [ ] Application starts locally without errors
- [ ] Can access http://localhost:5000
- [ ] Debug endpoint works: http://localhost:5000/api/debug/cameras

## 📱 Render Setup Checklist

### Account Setup
- [ ] Created Render account at https://render.com
- [ ] Email verified
- [ ] GitHub account connected to Render

### Web Service Configuration
- [ ] Service name set (e.g., "fallguard")
- [ ] Repository selected
- [ ] Branch set to "main"
- [ ] Build Command: `pip install -r requirements.txt`
- [ ] Start Command: `gunicorn --timeout 120 --workers 1 main:app`
- [ ] Region selected (recommended: Oregon)
- [ ] Plan selected (Free for testing, or Paid for production)

### Environment Variables (Optional)
- [ ] `TELEGRAM_BOT_TOKEN` added if using Telegram
- [ ] `ADMIN_PASSWORD` added if customizing
- [ ] Any other required env vars added

## 🚀 Deployment Checklist

### Before Clicking Deploy
- [ ] Read `RENDER_QUICKSTART.md`
- [ ] All requirements in checklist above are complete
- [ ] Render dashboard shows correct settings
- [ ] Final review of code on GitHub

### During Deployment
- [ ] Click "Create Web Service" in Render
- [ ] Monitor build logs in real-time
- [ ] Watch for any build errors
- [ ] Be patient (first deployment takes 2-5 minutes)

### After Deployment
- [ ] Render provides a public URL
- [ ] Build shows "✓ Build succeeded"
- [ ] Service shows "Live" status
- [ ] Can access the URL without errors

## 🧪 Post-Deployment Testing

### Functionality Tests
- [ ] Dashboard loads without errors
- [ ] API endpoints respond correctly
- [ ] Check logs: `https://dashboard.render.com → your-service → Logs`
- [ ] Debug info accessible: `https://your-url/api/debug/cameras`

### Common Issues Check
- [ ] Model loaded successfully (check logs)
- [ ] No port errors (Render assigns PORT automatically)
- [ ] No permission errors for file access
- [ ] Telegram integration working (if enabled)

### Performance Check
- [ ] First request completes (may be slow due to model loading)
- [ ] Subsequent requests are faster
- [ ] No memory issues in logs
- [ ] CPU usage is reasonable

## 📊 Ongoing Monitoring

### Daily Checks
- [ ] Service is showing "Live" status
- [ ] No errors in recent logs
- [ ] Response time is acceptable

### Weekly Checks
- [ ] Monitor service resource usage
- [ ] Check for any API errors
- [ ] Review Telegram notifications (if applicable)

### Monthly Checks
- [ ] Review and update dependencies if needed
- [ ] Check for security updates
- [ ] Consider performance optimization
- [ ] Verify backups of critical data

## 🔧 Troubleshooting Checklist

If something goes wrong:

- [ ] Check Render dashboard logs first
- [ ] Verify environment variables are set correctly
- [ ] Ensure model file is in Git repository
- [ ] Check if dependencies are all in requirements.txt
- [ ] Try redeploying (common quick fix)
- [ ] Review DEPLOY.md troubleshooting section
- [ ] Check Render status page for service issues

## 📝 Documentation Reference

Keep these files handy:
- [ ] `README_RENDER.md` - Overview
- [ ] `RENDER_QUICKSTART.md` - Quick reference
- [ ] `RENDER_SETUP.md` - Detailed setup
- [ ] `DEPLOY.md` - Comprehensive guide
- [ ] `.env.example` - Environment template

## 🎯 Success Criteria

Your deployment is successful when:
- ✅ Render shows "Live" status
- ✅ Public URL is accessible
- ✅ Dashboard loads without errors
- ✅ Logs show no critical errors
- ✅ Application responds to requests

---

## 💡 Pro Tips

1. **Start with Free Tier** - Deploy free first, upgrade later if needed
2. **Check Logs First** - 90% of issues visible in Render logs
3. **Enable Auto-Deploy** - Let Render redeploy on Git pushes
4. **Monitor Performance** - Watch resource usage, especially first week
5. **Keep Code Updated** - Push fixes quickly and redeploy

## 🆘 Need Help?

1. Check `DEPLOY.md` troubleshooting section
2. Review Render logs carefully
3. Visit https://render.com/docs
4. Contact Render support: support@render.com

---

**Good luck with your deployment! 🚀**
