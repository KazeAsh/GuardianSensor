from fastapi import APIRouter, HTTPException
from datetime import datetime
import psutil
import os
import logging

logger = logging.getLogger(__name__)
router = APIRouter(tags=["health"])

@router.get("/health")
async def health_check():
    """Comprehensive health check endpoint for CI/CD and monitoring"""
    try:
        # Basic system checks
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Application-specific checks
        data_dirs = [
            'data/raw/mmwave',
            'data/processed', 
            'outputs/visualizations'
        ]
        
        directory_status = {}
        missing_dirs = []
        for dir_path in data_dirs:
            exists = os.path.exists(dir_path)
            directory_status[dir_path] = exists
            if not exists:
                missing_dirs.append(dir_path)
        
        health_status = {
            "status": "healthy",
            "service": "GuardianSensor API",
            "version": "1.0.0",
            "timestamp": datetime.now().isoformat(),
            "system": {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "disk_percent": disk.percent,
                "uptime_seconds": psutil.boot_time()
            },
            "directories": directory_status,
            "environment": os.getenv('ENVIRONMENT', 'development')
        }
        
        # Check thresholds
        if cpu_percent > 80 or memory.percent > 85:
            health_status["status"] = "degraded"
            health_status["warning"] = "High resource usage"
        
        if disk.percent > 90:
            health_status["status"] = "degraded"
            if "warning" in health_status:
                health_status["warning"] += "; Low disk space"
            else:
                health_status["warning"] = "Low disk space"
        
        if missing_dirs:
            health_status["status"] = "degraded"
            health_status["missing_directories"] = missing_dirs
        
        return health_status
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=503, detail=f"Health check failed: {str(e)}")

@router.get("/health/ready")
async def readiness_probe():
    """Kubernetes readiness probe endpoint"""
    try:
        # Check required directories exist
        required_dirs = ['data', 'outputs']
        for dir_path in required_dirs:
            if not os.path.exists(dir_path):
                raise HTTPException(status_code=503, detail=f"Required directory missing: {dir_path}")
        
        return {"status": "ready"}
    except Exception as e:
        logger.error(f"Readiness check failed: {str(e)}")
        raise HTTPException(status_code=503, detail="Not ready")

@router.get("/health/live")
async def liveness_probe():
    """Kubernetes liveness probe endpoint"""
    return {"status": "alive", "timestamp": datetime.now().isoformat()}