# Machine Learning Projects

#### 1. Linear Regression
Housing Price Prediction using a Kaggle dataset. The processes done were:
  - Data Cleaning 
  - Data Visualization
  - Machine Learning Model Development (Scikit-learn Linear Regression, XGBoost Regressor Model, TensorFlow Neural Network Model) 
  - Explanations of Results

#### 2. Sentiment Analysis
Two Jupyter Notebooks were created. The first one is comparison of different models using Amazon Food Reviews dataset while the second is evaluation of IMDB dataset using the HuggingFace sentiment model. Also, new data from an unrelated topic (Resident Evil 4 Remake reviews) was evaluated using these models to check each model's performance.

#### 3. Speech-to-Text Transcriber with Sentiment Analysis Deployed using FastAPI
OpenAI's Whisper was used to transcribe three .wav files (2 Text-to-Speech reviews from the Resident Evil 4 Remake reviews and 1 snippet of the IGN's Hades Review). The HuggingFace Sentiment Analysis Model was used to evaluate the transcriptions from the audio files and predict its sentiment (POSITIVE, NEUTRAL, NEGATIVE). This was deployed in FastAPI and a Docker container was created.  

#### 4. Language Detection
AssemblyAI's Language Detection tutorial deployed in FastAPI, Docker and, instead of Heroku, it was deployed in AWS Lambda. AWS services used: ECR, IAM, Lambda.
