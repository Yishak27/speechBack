# Amharic Speech Recognition

## Setup

1. Install the dependencies:
    ```sh
    pip install -r requirements.txt
    ```

2. Run the Flask app:
    ```sh
    python app.py
    ```

## API Endpoints

### Train the Model

- **URL**: `/train`
- **Method**: `POST`
- **Payload**:
    ```json
    {
        "epochs": 20,
        "batch_size": 20
    }
    ```

### Test the Model

- **URL**: `/test`
- **Method**: `POST`
- **Payload**: Form-data with a key `file` containing the audio file.
