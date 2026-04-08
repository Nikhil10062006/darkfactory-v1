import uvicorn
import os
from server.main import app

def main():
    port = int(os.environ.get("PORT", 7860))
    host = os.environ.get("HOST", "0.0.0.0")
    uvicorn.run("server.main:app", host=host, port=port)

if __name__ == "__main__":
    main()
