# The process of learning and evaluation of results
To run the model training process, use a bash script as follows:
```
./run.sh
```
Be careful, this command will do everything for you, but the code execution will take a very long time (up to several days on some PC configurations).

For the command to work you need to use it in the Git Bash terminal. You can find the required terminal in Visual Studio Code, on the terminal bar:

![instruction_git_bash](https://github.com/artemisak/MicrobesAndGlucouseAnalysis/assets/76273674/c4298643-ee71-4bd5-9de0-9ed227f161c6)

# Process for rapid evaluation of results
In order to quickly reproduce the results of numerical experiments follow these steps:
1) Install Python 3.11
2) Create a virtual environment:
    ```
    python -m venv venv
    ```
3) Activate the virtual environment

    For Windows/Linux, go to the ```env/Scripts``` folder and run the script in the Powershell console
    ```
    ./activate
    ```
    If you use the Windows integrated command console instead of Powershell the activation will look like this:
    ```
    .\activate
    ```
    On a Mac, the procedure will be similar, but the directory you need will be at ``env/bin``.
    
    After that, go back to the directory where the files ```requirements.txt``` and ```fast_predict.py``` are located.

4) Install the necessary dependencies:
    ```
    python -m pip install -r requirements.txt
    ```

5) Run the evaluation code:

    ```
    python fast_predict.py
    ```
