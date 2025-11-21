# Starting the Server

## Quick Start

Use the startup script to ensure clean startup:

```bash
cd web_app
./start_server.sh
```

## Manual Start

If you prefer to start manually:

1. **Clean up any existing processes:**
   ```bash
   lsof -ti:5003 | xargs kill -9
   pkill -9 -f "python.*app.py"
   ```

2. **Activate virtual environment:**
   ```bash
   source ../venv/bin/activate
   ```

3. **Start the server:**
   ```bash
   cd web_app
   python3 app.py
   ```

## Troubleshooting

If the page loads forever:

1. **Check for stopped processes:**
   ```bash
   ps aux | grep "python.*app.py"
   ```
   Kill any stopped (T status) processes:
   ```bash
   pkill -9 -f "python.*app.py"
   ```

2. **Check port availability:**
   ```bash
   lsof -i :5003
   ```
   Kill any process using port 5003:
   ```bash
   lsof -ti:5003 | xargs kill -9
   ```

3. **Clear browser cache** and try again

4. **Check server logs:**
   ```bash
   tail -f web_app/server.log
   ```

## Server Features

- Auto-reload enabled (restarts on code changes)
- Threaded mode for better concurrency
- Runs on http://127.0.0.1:5003
