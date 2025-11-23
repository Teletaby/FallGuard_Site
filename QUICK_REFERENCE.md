# 🚀 RENDER DEPLOYMENT REFERENCE CARD

## One-Page Quick Reference

---

## ⚡ 3-Step Deployment (10 minutes)

### Step 1: Git Push
```bash
cd FallGuard_test-main
git add .
git commit -m "Deploy to Render"
git push origin main
```

### Step 2: Create Service
1. Go to: https://dashboard.render.com
2. Click: "New +" → "Web Service"
3. Select: Your repository
4. **Start Command**: `gunicorn --timeout 120 --workers 1 main:app`
5. Click: "Create Web Service"

### Step 3: Live! 🎉
- Wait 2-5 minutes
- Get public URL
- Done!

---

## 📚 Documentation Quick Links

| Need | Read |
|------|------|
| Quick start | `START_HERE.md` |
| Step-by-step | `RENDER_QUICKSTART.md` |
| Full guide | `DEPLOY.md` |
| Checklist | `DEPLOYMENT_CHECKLIST.md` |
| All details | `RENDER_SETUP.md` |

---

## 🔧 Configuration Reference

### Procfile Content
```
web: gunicorn --timeout 120 --workers 1 main:app
```

### runtime.txt Content
```
python-3.10.13
```

### Environment Variables (Optional)
```
TELEGRAM_BOT_TOKEN=your_token
ADMIN_PASSWORD=your_password
```

---

## 🎯 What's Configured

✅ Port: Automatic (from PORT env var)
✅ Python: 3.10.13
✅ Startup: Gunicorn with 120s timeout
✅ Workers: 1 (memory efficient)
✅ Auto-deploy: On Git push

---

## 💻 Local Testing

**Windows:**
```
.\setup_local.bat
python main.py
```

**Mac/Linux:**
```
bash setup_local.sh
python main.py
```

Visit: http://localhost:5000

---

## ⚠️ Important Notes

- **Free Tier**: Services sleep after 15 min
- **Paid Tier**: Always running ($7+/month)
- **Files**: No persistent storage (use cloud)
- **Camera**: Use video streams, not webcam

---

## 🆘 Troubleshooting

| Problem | Fix |
|---------|-----|
| Build fails | Check Render logs |
| Model not found | Verify in Git |
| Port error | Already fixed ✅ |
| Slow start | Model loading time |

---

## 📍 Key Files

```
Procfile ..................... Startup config
runtime.txt .................. Python version
main.py ...................... Updated for Render
requirements.txt ............. Dependencies
render.yaml .................. Advanced config
```

---

## 🎯 Success = 

✅ Render shows "Live"
✅ Public URL accessible
✅ Dashboard loads
✅ No critical errors in logs

---

## 📞 Help

- **Logs**: https://dashboard.render.com → Logs tab
- **Docs**: `DEPLOY.md` (troubleshooting section)
- **Render**: https://render.com/docs

---

## 🎊 Status: READY TO DEPLOY! 🎊

**Next:** Read `START_HERE.md` and deploy!

---

*Print this card for quick reference during deployment*
