## What is PredictAd?

***PredictAd*** is a application build for Youtube AD analysis.

It take in input a Youtube Ad and provide in output the scores and some relevent recommendations to improve the Ad. Curious to see the architecture? It's [here:](https://github.com/Hippolyte90/PredictAd/blob/main/PredictAd%20ARCHITECTURE.pdf)

## How to compute this app?

- Install Git
- Clone my git repository 
  
  ```powershell
   git clone [URL_OF_THIS_REPOSITORY]
  ```
- Create a new virtual environment
  
 ```powershell
   conda create -n <your ENV name> python=3.12 -c conda-forge
   conda activate <your ENV name>
 ```

- Install relevent dependence
  
  ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
  ```
- Create a file name ***".env"*** in your folder and put in you ***HuggingFace key*** and your ***OpenAI Key***.
  
- Compute directly the file in **Git bash** or **bash(MSYS2)**
  
  ```bash
  predictad_app.py
  ```