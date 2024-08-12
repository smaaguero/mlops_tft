# PH Teamfight Tactics Data Analysis - MLOps Enhanced Version

## Introduction

This project originally involved building an ETL pipeline to retrieve and analyze match data of PH Challenger/GM+ players for the game Teamfight Tactics (TFT) using the Riot Developer API. The analysis aimed to predict the in-game placement of players based on their combinations of units, traits, and items. The current iteration of the project introduces MLOps best practices to ensure a more scalable, maintainable, and robust machine learning pipeline. 

## Enhancements and Modifications

### 1. **Riot API Integration Update**
   - **Issue**: The original ETL pipeline broke due to changes in the Riot Developer API.
   - **Fix**: Updated the API integration to match the latest specifications, ensuring data collection processes are functional and accurate.

### 2. **Pipeline Framework Implementation (Kedro)**
   - **Enhancement**: Migrated the entire project to Kedro, a popular pipeline framework, to better organize, modularize, and manage the data and machine learning workflows.
   - **Benefit**: Kedro provides a clear structure for the data science workflow, allowing for easier debugging, testing, and versioning. It also ensures that the pipeline is reproducible and scalable.

### 3. **Data and Model Version Control**
   - **Enhancement**: Integrated Data Version Control and Model Version Control to handle data and model versioning. All datasets and models are now version-controlled.
   - **Benefit**: This ensures that changes to datasets or models are tracked over time, making it easier to revert to previous versions and maintain reproducibility. DVC also facilitates collaboration by enabling consistent data and model management across different environments. The model versioning allows for comparison between different versions of the model, ensuring that the best-performing model is always in production. It also facilitates the rollback to previous models in case of performance degradation.

### 4.  **Dependency Management with Poetry**
   - **Enhancement**:Migrated the project’s dependency management from a traditional requirements.txt approach to Poetry.
   - **Benefit**: Poetry simplifies dependency management by automatically resolving and locking dependencies, ensuring consistency across different environments. It also manages virtual environments, making it easier to handle different project environments.
### 5. **Python Version Management with Pyenv**
   - **Enhancement**: Integrated pyenv for managing Python versions within the project.
   - **Benefit**: pyenv allows for easy switching between different Python versions, ensuring that the project can be developed and tested across multiple Python versions if needed. It also ensures that the correct Python version is used across different environments, reducing the likelihood of compatibility issues.

## **In Work**

### 6. **Automated Testing and Continuous Integration**
   - **Enhancement**: Introduced unit tests, integration tests, and end-to-end tests across the pipeline using `pytest` and integrated these with a CI/CD pipeline using GitHub Actions.
   - **Benefit**: Automated testing ensures that any changes to the codebase do not introduce bugs or break existing functionality. The CI/CD pipeline automatically tests and deploys the code, improving development speed and reliability.

### 7. **Improved Code Documentation and Encapsulation**
   - **Enhancement**: Refactored the codebase to improve encapsulation, modularity, and readability. Comprehensive docstrings were added to all functions and classes.
   - **Benefit**: Better code organization and documentation make the project easier to maintain and extend, especially for new contributors.

### 8. **ETL Process Enhancement**
   - **Enhancement**: The ETL process was revamped to handle the latest API changes and optimized for performance and reliability.
   - **Benefit**: The data extraction, transformation, and loading processes are now faster and more resilient to API changes, ensuring consistent data availability for analysis.

## Conclusion

The enhancements made to the original project align with MLOps best practices, making the pipeline more robust, scalable, and maintainable. By incorporating tools like Kedro, DVC, and automated testing, this project is now well-equipped for continuous improvement and deployment in real-world scenarios. The inclusion of model and data versioning further strengthens the reliability and reproducibility of the analysis and predictions.

## Results and Future Work

The updated pipeline and model show improved accuracy and robustness. Future work could involve expanding the analysis to include data from multiple regions or incorporating more advanced machine learning techniques, such as deep learning models, to enhance prediction accuracy.