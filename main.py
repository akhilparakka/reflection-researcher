import uvicorn
from dotenv import load_dotenv

load_dotenv()


def main():
    uvicorn.run("src.server:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    main()
