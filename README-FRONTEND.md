# 🚀 Space Invaders DQN Frontend

A modern web interface for the Space Invaders Deep Q-Learning (DQN) project. This frontend provides an intuitive dashboard to train, test, and visualize your AI agent's performance in real-time.

## ✨ Features

### 🎮 **Interactive Dashboard**
- Real-time training progress monitoring
- Performance metrics and statistics
- Beautiful data visualizations
- Responsive design for all devices

### 🧠 **Training Center**
- Start/stop training with custom parameters
- Live progress tracking
- Training statistics and charts
- Model weight management

### 🧪 **Testing Suite**
- Run performance evaluations
- Detailed test results and analysis
- Episode-by-episode breakdown
- Performance comparison tools

### 🎬 **Live Demo**
- Real-time game visualization
- Watch AI agent play Space Invaders
- Step-by-step action analysis
- Interactive controls (play, pause, reset)

### 📊 **Model Information**
- Complete model architecture details
- Layer-by-layer breakdown
- Parameter counts and configurations
- Weight management (load/save)

## 🏗️ Architecture

```
Frontend (React)          Backend (Flask)           ML Model (TensorFlow)
┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐
│   Dashboard     │      │   REST API      │      │   DQN Agent     │
│   Training      │◄────►│   Endpoints     │◄────►│   Environment   │
│   Testing       │      │   WebSocket     │      │   Preprocessing │
│   Demo          │      │   File Upload   │      │   Visualization │
│   Model Info    │      │   Real-time     │      │   Training      │
└─────────────────┘      └─────────────────┘      └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- **Python 3.8+** with pip
- **Node.js 16+** with npm
- **Git** for cloning the repository

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd space_invader_rl
   ```

2. **Install dependencies:**
   ```bash
   # Install Python dependencies
   pip install -r requirements-frontend.txt
   
   # Install Node.js dependencies
   npm install
   ```

3. **Start the application:**
   
   **On Windows:**
   ```cmd
   start.bat
   ```
   
   **On Linux/Mac:**
   ```bash
   ./start.sh
   ```
   
   **Manual start:**
   ```bash
   # Terminal 1 - Backend
   python app.py
   
   # Terminal 2 - Frontend
   npm start
   ```

4. **Access the application:**
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:5000

## 🐳 Docker Deployment

### Using Docker Compose

1. **Build and start all services:**
   ```bash
   docker-compose up --build
   ```

2. **Access the application:**
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:5000
   - Nginx (Production): http://localhost:80

### Individual Docker Services

**Backend only:**
```bash
docker build -f Dockerfile.backend -t space-invaders-backend .
docker run -p 5000:5000 space-invaders-backend
```

**Frontend only:**
```bash
docker build -f Dockerfile.frontend -t space-invaders-frontend .
docker run -p 3000:3000 space-invaders-frontend
```

## 📱 Usage Guide

### 1. **Dashboard**
- View overall system status
- Monitor training progress
- Access quick actions
- Check model information

### 2. **Training**
- Click "Setup Environment" to initialize
- Configure training parameters
- Start/stop training sessions
- Monitor real-time progress

### 3. **Testing**
- Set number of test episodes
- Run performance evaluations
- View detailed results
- Analyze agent performance

### 4. **Demo**
- Watch AI agent play live
- Use controls to play/pause/reset
- View real-time game state
- Analyze agent decisions

### 5. **Model Info**
- View model architecture
- Check parameter counts
- Manage model weights
- Load/save trained models

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the root directory:

```env
# Backend Configuration
FLASK_ENV=development
FLASK_DEBUG=True
API_HOST=0.0.0.0
API_PORT=5000

# Frontend Configuration
REACT_APP_API_URL=http://localhost:5000
REACT_APP_WS_URL=ws://localhost:5000

# Model Configuration
MODEL_PATH=./models
WEIGHTS_PATH=./models/weights
```

### Training Parameters

Modify training settings in the Training page:
- **Total Steps**: Number of training steps (default: 100,000)
- **Learning Rate**: Model learning rate (default: 1e-4)
- **Memory Size**: Experience replay buffer size (default: 1,000,000)
- **Epsilon Range**: Exploration rate range (default: 1.0 → 0.1)

## 📊 API Endpoints

### Training Endpoints
- `POST /api/train/start` - Start training
- `POST /api/train/stop` - Stop training
- `GET /api/train/status` - Get training status

### Testing Endpoints
- `POST /api/test/run` - Run performance test
- `GET /api/test/results` - Get test results

### Demo Endpoints
- `POST /api/demo/step` - Run single demo step
- `POST /api/demo/reset` - Reset demo environment

### Model Endpoints
- `GET /api/model/info` - Get model information
- `POST /api/weights/load` - Load model weights
- `POST /api/weights/save` - Save model weights

### Utility Endpoints
- `GET /api/health` - Health check
- `POST /api/setup` - Setup environment
- `GET /api/plots/training` - Get training plots

## 🎨 Customization

### Styling
The frontend uses styled-components for styling. Main theme colors:
- Primary: `#00d4ff` (Cyan)
- Success: `#2ed573` (Green)
- Warning: `#ffa502` (Orange)
- Danger: `#ff4757` (Red)

### Adding New Features
1. Create new components in `src/components/`
2. Add new pages in `src/pages/`
3. Update routing in `src/App.js`
4. Add API endpoints in `app.py`

## 🐛 Troubleshooting

### Common Issues

1. **Backend not starting:**
   ```bash
   # Check Python dependencies
   pip install -r requirements-frontend.txt
   
   # Check if port 5000 is available
   netstat -an | grep 5000
   ```

2. **Frontend not loading:**
   ```bash
   # Clear npm cache
   npm cache clean --force
   
   # Reinstall dependencies
   rm -rf node_modules package-lock.json
   npm install
   ```

3. **Model not loading:**
   - Ensure model weights file exists
   - Check file permissions
   - Verify model architecture compatibility

4. **CORS errors:**
   - Check Flask-CORS configuration
   - Verify API URL in frontend
   - Check browser console for errors

### Performance Issues

1. **Slow training:**
   - Use GPU acceleration if available
   - Reduce batch size
   - Decrease model complexity

2. **High memory usage:**
   - Reduce experience replay buffer size
   - Use smaller model architecture
   - Close unnecessary browser tabs

## 📈 Performance Monitoring

### Backend Monitoring
- Check Flask logs for errors
- Monitor memory usage during training
- Verify API response times

### Frontend Monitoring
- Use browser DevTools
- Check Network tab for API calls
- Monitor Console for JavaScript errors

## 🔒 Security Considerations

- Change default API keys in production
- Use HTTPS in production environments
- Implement proper authentication if needed
- Validate all user inputs

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **React** - Frontend framework
- **Flask** - Backend API framework
- **TensorFlow/Keras** - Machine learning framework
- **OpenAI Gym** - Reinforcement learning environment
- **Styled Components** - CSS-in-JS styling

## 📞 Support

For support and questions:
- Create an issue on GitHub
- Check the troubleshooting section
- Review the API documentation

---

**Happy Training! 🚀**

*Train your AI agent to become a Space Invaders champion with this beautiful web interface!*

