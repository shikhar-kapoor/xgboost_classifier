# xgboost_classifier
xgboost-training-and-containerization

Guide: Containerizing ML Models for Deployment
This guide contains info on how to package the following ML Model assets into Docker containers, so the ML Model can be deployed into production:
1. Code Organization & Version Control:
- Structure your model code into a clear, modular format and commit it to a version-controlled repository (e.g., GitHub).
- See file name = 1_train_xgb.py
2. Environment & Dependency Management:
- Specify and lock down all required libraries and versions (e.g., using requirements.txt or Pipfile).
o IMPORTANT!: The libraries/dependencies included must be for all training, prediction & API service codes
- You can install all dependencies by running: pip install -r requirements.txt
- See file name = 2_requirements.txt
3. Model Artefact Serialization:
- Save the trained model in a stable, portable format (e.g., using joblib or pickle).
o This usually done by the data scientists in the same Jupyter notebook that they trained the model in;
this enables Data Scientists to save the trained model for future use, so you don’t need to re-train it every time
- See file name = 3_serialising_model.py
4. Predict Function
- Create a predict function to:
o Test that the prediction pre-processing is working on new data
o Test that the model (pickle file) is predicting the target variable
o Ensure that the prediction output is in your desired format (see #Print Predictions section in the file below)
- See file name = 4_predict_function.py
5. API Interface & Integration:
- Wrap your model inference logic in a simple API (e.g., using Flask or FastAPI) so that the model can be served via HTTP requests.
- Simply ask ChatGPT to wrap your a) Pickle file & b) Prediction Function into an API serivce using one of the following frameworks below.
- There are many frameworks to wrap a model into an API: Options are:
o Flask Microservice
A lightweight Python web framework that can easily wrap your model loading and prediction logic in simple route endpoints.
Quick to set up and suitable for simple APIs.
o Fast API
A modern, fast (asynchronous) web framework that simplifies the creation of RESTful APIs.
Automatic documentation via OpenAPI and great performance.
More structured than Flask, making it easy to define request/response models and integrate with other tools.
Recommendation:
- Flask: For small scale projects, quick prototypes or when simplicity is ok
- FastAPI: More complex, modern APIs that require performance, scalability & built-in validation / documentation
o Django / Other full stack frameworks
Heavier, more feature-rich frameworks for building full web applications.
Typically more than you need if you only want a prediction endpoint.
o Serverless Functions (AWS Lambda, G Cloud Functions, Azure Functions)
Deploy your model inference code as a function in the cloud.
Scaling and infrastructure are managed for you, but you’ll need to handle environment setup and possibly package
dependencies in a specific way.
o API Gateways or Managed ML Platforms
Use managed services like AWS SageMaker or GCP Vertex AI to host your model and create endpoints automatically.
Less code to maintain, but involves platform lock-in and potentially higher cost.
5.1 Testing the API (through Terminal)
1. Files organized in the same folder (directory) as follows:
2. 3. 4. 5. 6. Run the following command to navigate the Terminal to your directory
 cd "/Users/shikhar/Documents/Ship2MyID/MLOps - App>Ser>Dock>K8S/Recommender Service/"
Install dependencies by running the following command:
 pip install -r requirements.txt
Run uvicorn (start the server)
 uvicorn main:app –reload
Once Uvicorn server is up:
 Open a NEW Terminal window
 Run: pip install -r requirements.txt
 Run the cURL command to test the API
 curl -X POST "http://127.0.0.1:8000/predict" \
 -H "Content-Type: application/json" \
 -d '{"Deal_ID": "D001","User_ID": "U1001","Deal_Category":
"Percent_20","Product_Category":"Pizza","Day_Of_Week_Start":"Wednesday","Day_Of_Week_End":"Wednesday","Deal_
Duration_Hours":2.0}'
If Successful, you will receive the response
6. Containerization (Docker):
- Package the code, dependencies, and model artifact into a Docker container for consistent deployment across environments.
- Package your API (Flask/FastAPI) and model into a Docker image.
- Deploy that image to a container orchestrator or service.
Steps:
1. Organize your project directory:
a. ├── main.py # Your FastAPI application
b. ├── xgb_model.pkl # Serialized model
c. ├── requirements.txt # Dependencies
d. ├── Dockerfile # Dockerfile for containerization
2. Write a Dockerfile to define the container image. - the file has no extension, it should just be named Dockerfile
a. See sample in '/Users/shikhar/Documents/Ship2MyID/MLOps - App>Ser>Dock>K8S/6_Dockerfile'
b. For another sample of manual dependency installation from Muthu (Ship2MyID) see: '/Users/shikhar/Documents/Ship2MyID/MLOps -
App>Ser>Dock>K8S/Muthu_Dockerfile'
3. Build the Docker image.
a. Run command in Terminal: docker build -t fastapi-recommender-service .
b. This will package your code, dependencies, and model artifact into a Docker image named fastapi-recommender-service.
4. Run the container
a. Run command in Terminal: docker run -d -p 8000:8000 fastapi-recommender-service
5. Test the container
a. Visit on browser or use postman to hit the endpoint: http://127.0.0.1:8000/docs
b. This should display the FastAPI Swagger UI, confirming your app is running inside the container.
6. Debugging The Container
a. To view logs from the container, run in Terminal:
i. docker logs <container_id>
b. To access the container’s shell:
i. docker exec -it <container_id> /bin/bash
7. Push the image to a Docker registry for deployment. If you want to deploy the container on a server or Kubernetes cluster, push it to a
Docker registry (e.g., Docker Hub, Azure Container Registry, AWS ECR, etc.)
