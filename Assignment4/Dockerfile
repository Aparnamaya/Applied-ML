# Use official Python image as the base
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy all necessary files
COPY app.py score.py best_model.pkl vectorizer.pkl r.txt ./

# Install dependencies
RUN pip install --no-cache-dir -r r.txt

# Expose the port your app runs on
EXPOSE 5000

# Start the Flask app
CMD ["python", "app.py"]
