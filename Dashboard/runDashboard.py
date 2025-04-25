# run_dashboard.py (or any other name)
from Dashboard.app import app
from Dashboard.callbacks import register_callbacks
from Dashboard.layout import create_layout
from Dashboard.utils import get_available_books

# Load initial data for layout
available_books = get_available_books()
initial_book_name = available_books[0] if available_books else None

app.layout = create_layout(available_books, initial_book_name)
register_callbacks(app)


if __name__ == "__main__":
    app.run_server(debug=True)
