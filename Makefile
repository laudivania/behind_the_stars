# Instal requirements
install_requirements:
	@pip install -r requirements.txt

reinstall_package:
	@pip uninstall -y taxifare || :
	@pip install -e .

#Streamlit
streamlit:
	-@streamlit run app.py

#API
run_api:
	uvicorn behind_the_stars.api.fast:app --reload

#Install the Python package contained in the current folder and update it if it is already installed.
install:
	@pip install . -U

#Cleaning
clean:
	@rm -fr */__pycache__
	@rm -fr __init__.py
	@rm -fr build
	@rm -fr dist
	@rm -fr *.dist-info
	@rm -fr *.egg-info
	-@rm model.joblib
