import subprocess
import sys
import time

def main():
    print("="*60)
    print("[ STARTING ] FairCredit Africa Services...")
    print("="*60)
    
    # Run all as child processes pointing to the same venv
    try:
        print("1. Starting MLflow Tracking Server (PORT 5000)")
        mlflow_proc = subprocess.Popen(
            [sys.executable, "-m", "mlflow", "server", "--host", "127.0.0.1", "--port", "5000"],
            stdout=sys.stdout,
            stderr=sys.stderr
        )
        time.sleep(2)
        
        print("2. Starting FastAPI Backend (PORT 8000)")
        fastapi_proc = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "fastapi_app:app", "--host", "127.0.0.1", "--port", "8000", "--reload"],
            stdout=sys.stdout,
            stderr=sys.stderr
        )
        time.sleep(2)
        
        print("3. Starting Streamlit Frontend (PORT 8501)")
        streamlit_proc = subprocess.Popen(
            [sys.executable, "-m", "streamlit", "run", "streamlit_app.py"],
            stdout=sys.stdout,
            stderr=sys.stderr
        )
        
        print("\n" + "="*60)
        print("[ SUCCESS ] All services are running!")
        print("   - Streamlit Application: http://localhost:8501")
        print("   - FastAPI Swagger Docs:  http://localhost:8000/docs")
        print("   - MLflow Dashboard:      http://localhost:5000")
        print("="*60)
        print("Press Ctrl+C to stop all services.\n")

        # Keep alive and monitor
        while True:
            time.sleep(1)
            if mlflow_proc.poll() is not None:
                print("MLflow crashed!")
                break
            if fastapi_proc.poll() is not None:
                print("FastAPI crashed!")
                break
            if streamlit_proc.poll() is not None:
                print("Streamlit crashed!")
                break

    except KeyboardInterrupt:
        print("\n[ STOPPING ] Shutting down all services gracefully...")
    
    finally:
        # Guarantee cleanup
        if 'mlflow_proc' in locals() and mlflow_proc.poll() is None:
            mlflow_proc.terminate()
        if 'fastapi_proc' in locals() and fastapi_proc.poll() is None:
            fastapi_proc.terminate()
        if 'streamlit_proc' in locals() and streamlit_proc.poll() is None:
            streamlit_proc.terminate()
            
        print("System shutdown complete.")

if __name__ == "__main__":
    main()
