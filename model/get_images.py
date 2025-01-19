import kagglehub

# Download latest version
path = kagglehub.dataset_download("paulchambaz/google-street-view")

print("Path to dataset files:", path)