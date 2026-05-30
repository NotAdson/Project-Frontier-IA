# Web Interface

This directory contains a standalone Flask web application that allows a human player to battle against the backend AI agent in a retro GameBoy Advance style interface.

## Structure
- **`app.py`**: The Flask backend server that interfaces with the `PokemonProblem` and the Battle Agent, parsing game states to send to the frontend.
- **`static/`**: Contains the CSS (GameBoy Advance styling) and JavaScript (frontend logic and Showdown sprite fetching).
- **`templates/`**: Contains the `index.html` structure.

## Usage
Run the app:
```bash
python3 src/web/app.py
```
Then open your browser to `http://localhost:5000`.
