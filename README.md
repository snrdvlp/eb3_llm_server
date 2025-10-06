# eb3_llm_server
Provides a chat endpoint for eb3 insurance llm

# Command to run the server
uvicorn llm_server:app --host 0.0.0.0 --port 8000 --reload


sudo nano /etc/systemd/system/llm_server.service