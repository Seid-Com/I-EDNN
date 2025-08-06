# I-EDNN Deployment Guide

This guide covers multiple deployment options for the I-EDNN (Ising-Enhanced Deep Neural Network) application.

## Quick Deployment Options

### 1. GitHub Repository Setup

1. **Create a new GitHub repository**:
   ```bash
   git init
   git add .
   git commit -m "Initial commit: I-EDNN physics-inspired ML framework"
   git branch -M main
   git remote add origin https://github.com/seidmehammed/i-ednn.git
   git push -u origin main
   ```

2. **Enable GitHub Actions** (automatic):
   - GitHub Actions workflow is already configured in `.github/workflows/deploy.yml`
   - Will automatically run tests and deploy documentation on push

### 2. Streamlit Cloud Deployment

1. **Connect to GitHub**:
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Connect your GitHub account
   - Select your `i-ednn` repository

2. **Configuration**:
   - Main file path: `app.py`
   - Python version: `3.11`
   - Dependencies will be automatically installed from `setup.py`

3. **Deploy**: 
   - Click "Deploy"
   - Your app will be available at `https://seidmehammed-i-ednn-app-xyz.streamlit.app`

### 3. Docker Deployment

1. **Build the image**:
   ```bash
   docker build -t i-ednn .
   ```

2. **Run locally**:
   ```bash
   docker run -p 8501:8501 i-ednn
   ```

3. **Deploy to cloud**:
   - **Google Cloud Run**:
     ```bash
     gcloud builds submit --tag gcr.io/PROJECT_ID/i-ednn
     gcloud run deploy --image gcr.io/PROJECT_ID/i-ednn --platform managed
     ```
   
   - **AWS ECS** or **Azure Container Instances**: Use the Docker image with your preferred container service

### 4. Replit Deployment (Current Environment)

Your app is already running on Replit! To deploy:

1. **Click the Deploy button** in the Replit interface
2. **Choose deployment type**:
   - Static deployment for documentation
   - Autoscale deployment for full application
3. **Configure domain** (optional): Set up custom domain
4. **Deploy**: Your app will be available at `https://your-app-name.yourusername.repl.co`

## Advanced Deployment

### Heroku Deployment

1. **Create Procfile**:
   ```
   web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
   ```

2. **Deploy**:
   ```bash
   heroku create your-app-name
   git push heroku main
   ```

### Railway Deployment

1. **Connect GitHub repository** at [railway.app](https://railway.app)
2. **Configure**: Railway will auto-detect Streamlit and deploy
3. **Environment**: No additional configuration needed

### Local Development

1. **Clone and setup**:
   ```bash
   git clone https://github.com/seidmehammed/i-ednn.git
   cd i-ednn
   pip install -e .
   ```

2. **Run application**:
   ```bash
   streamlit run app.py
   ```

3. **Access**: Open `http://localhost:8501`

## Environment Variables

For production deployments, you may want to set:

```bash
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
STREAMLIT_SERVER_HEADLESS=true
STREAMLIT_SERVER_ENABLE_CORS=false
STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false
```

## Performance Optimization

For production deployments:

1. **Enable caching**: Already implemented in the app with `@st.cache_data`
2. **Optimize model size**: Consider model quantization for faster inference
3. **Resource limits**: Set appropriate CPU/memory limits in container deployments
4. **CDN**: Use CDN for static assets if deploying to cloud platforms

## Monitoring

- **Streamlit Cloud**: Built-in monitoring and logs
- **Docker**: Use container monitoring tools like Prometheus/Grafana
- **Cloud platforms**: Native monitoring services (CloudWatch, Stackdriver, etc.)

## Security

- **HTTPS**: Automatically enabled on most platforms
- **Environment secrets**: Use platform-specific secret management
- **CORS**: Configured for security in production

## Troubleshooting

**Common issues**:
- **Port conflicts**: Ensure port 8501 is available
- **Memory issues**: Increase container memory for large datasets
- **Dependency conflicts**: Use the exact versions specified in `setup.py`

**Platform-specific**:
- **Streamlit Cloud**: Check app logs in dashboard
- **Docker**: Use `docker logs container-name`
- **Heroku**: Use `heroku logs --tail`

## Support

For deployment help:
- **Documentation**: Check platform-specific documentation
- **GitHub Issues**: Create an issue in the repository
- **Community**: Streamlit community forum for Streamlit-specific issues

---

Choose the deployment method that best fits your needs. Streamlit Cloud is recommended for quick sharing, while Docker provides maximum flexibility for production environments.