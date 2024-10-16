# Project-2
> [!NOTE]
> This project requires Python version 3.6.9+. Using a Python version lower than 3.6.9 may cause dependencies to become corrupted or fail to install properly.
>
> The `python` command may not be recognised because it wasn’t set as an alias for `python3` during installation. If your system doesn’t recognise the `python` command, use `python3` instead. Similarly, if `pip` isn’t recognised, replace it with `pip3`.

### Installation and setting up the project
Follow these steps to set up project:
1. Clone the repository
   ```sh
   git clone https://github.com/DuckyShine004/chat-app-python
   ```
2. Create a virtual environment (ensure you're using Python 3.6.9 or later)
   ```sh
   python -m venv .venv
   ```
3. Activate the virtual environment:
   ```sh
   source ./vevn/bin/activate
   ```
   On Windows:
   ```sh
   source .venv\Scripts\activate
   ```
4. Before installing dependencies, upgrade pip, setuptools, and wheel by running
   ```sh
   pip install --upgrade pip setuptools wheel
   ```
5. Install the required dependencies:
   ```sh
   pip install -r requirements.txt
   ```

### Running the project
Follow these chronological steps (within the root directory) to run the project:
1. Load data into a data frame:
   ```sh
   python -m src.utilities.ConvertToDataFrame
   ```
2. Retrieve the best model parameters:
   ```sh
   python -m src.SVM.SVMTuning
   ```
3. Test the SVM:
   ```sh
   python -m src.SVM.testSVM && python -m src.SVM.testSVMShowFailingImage
   ```
4. Create the final model to load onto the jetbot:
   ```sh
   python -m src.FinalModelCreation
   ```
