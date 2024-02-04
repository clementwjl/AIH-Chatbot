# AIH-Chatbot
Telegram Chat Bot boilerplate code for AIH, with a script to evaluate the RAG pipeline using Ragas.

As part of your project, you are required to implement a chatbot using Langchain. However, implementing a whole new web application may be too tedious and out-of-scope, so we have provided code to implement your code to a telegram bot instead.

## Part 1: Setting up your Telegram Bot
## What to do: 

### 1. Clone to your repository

On your preferred IDE, open the folder that you wish you put the project in, and proceed to run the following in your shell:

```
git clone https://github.com/clementwjl/AIH-Chatbot.git
```

And afterwhich,

```
cd AIH-Chatbot
```

### 2. Creating the environment variables

On the `AIH-Chatbot` directory, create a `.env` file or use an existing one (if you have from the lab). Open the file to the following:

```
OPENAI_API_KEY=sk-YOUR_OPENAI_APIKEY
LANGSMITH_API_KEY=ls__YOUR_LS_APIKEY
TELEGRAM_BOT_TOKEN=YOUR_BOTTOKEN
```

Use your group's OpenAI and LangSmith API Key.

We will talk about how to get your Telegram bot token in the next step.

### 3. Get your telegram API key

You will need the API key to connect to a bot. This requires you to navigate to [BotFather](https://t.me/BotFather). Do refer to [this video](https://www.youtube.com/watch?v=aNmRNjME6mE&ab_channel=SmartBotsLand) should you need help to gather the API key.

After you receive the API key, save it into the `.env` file.

### 4. Install all dependencies

Windows: 
Run the following on your terminal (Command Prompt)

```
pip install -r requirements.txt
```

Mac: Run the following on your terminal (zsh)
```
pip3 install -r requirements.txt
```

### 5. Loading your source documents
Before running the evaluation script, make sure to create a folder named 'docs' in the same repository. Deposit PDF source documents into this folder. A sample test document has been provided in the repository for initial testing.

Once done, your folder should look like this:

![Screenshot 2024-02-05 at 12 57 31 AM](https://github.com/clementwjl/AIH-Chatbot/assets/108287396/89c38110-07c6-45f6-9c51-92ce7bbefad0)


### 6. How to make your code run with the bot

There are two Python files that will be running the show, `bot.py` and `model.py`.

- `bot.py` will assist in receiving and sending out responses. 

- `model.py` currently consists of only one function, `getResponses(question)`. It takes in the user's input and should return the message that we would like to return to the user. You are to modify this function with your current model. 

You may use helper functions or even change `bot.py` to suit your project requirements. Ultimately, it is up to your group to decide how the chatbot should behave. 

## 7. Test the bot

Windows: Run the following on your terminal (Command Prompt)
```
python bot.py
```

Mac: Run the following on your terminal (zsh)
```
python3 bot.py
```

If everything works, it should produce the following:
```
Loading configuration...
Successfully loaded! Starting bot...
```

Head to your Telegram bot and give it a test. 


## Error debugging

My terminal says
```
 File "/opt/homebrew/lib/python3.11/site-packages/chromadb/api/types.py", line 99, in maybe_cast_one_to_many
    if isinstance(target[0], (int, float)):
                  ~~~~~~^^^
IndexError: list index out of range
```
A: Create a folder called docs, add all your relevant documents inside. 

## Part 2: How to evaluate your RAG pipeline with Ragas

As part of the Langchain framework, the _evaluator.py_ script helps with assessing the performance of your chatbot's RAG pipeline. This script leverages the RAGAS framework for evaluations, providing insights into various metrics such as context precision, faithfulness, answer relevancy, and context recall.

## How it works
How it Works
### 1. Load Source Documents
Before running the evaluation script, make sure to create a folder named docs in the same repository. Deposit PDF source documents into this folder. A sample test document has been provided in the repository for initial testing.

### 2. Create Vector Store
The script starts by creating a vector store with embeddings using the source documents stored in the docs folder.

### 3. Generate Evaluation Dataset
The script utilizes the source documents and the LLM defined to generate a small dataset. This dataset includes questions, contexts, and ground truth answers, forming the basis for evaluating the model's performance later on.

### 4. Generate Answers
Using the dataset, the script generates answers using the specified LLM. The resulting dataset now contains the original fields (questions, contexts, ground truth answers) along with the additional field of the generated answers by the LLM.

### 5. Apply RAGAS Framework
The script employs the RAGAS framework and package to evaluate how the model responses performed against the initial ground truths. The evaluation metrics include context precision, faithfulness, answer relevancy, and context recall.

### Running the Script
_***Note that the execution of this script might take awhile. **_

Execute the following command in your terminal to run the evaluator.py script:

Windows: Run the following on your terminal (Command Prompt)
```
python evaluator.py
```

Mac: Run the following on your terminal (zsh)
```
python3 evaluator.py
```

Upon successful execution, you will observe two things:
#### 1. Your folder will be populated with 2 new CSVs, titled _groundtruth_eval_dataset.csv_ and _basic_qa_ragas_dataset.csv_ respectively.
#### 2. the script will produce a terminal print statement containing evaluation metrics. An example output may look like the following:

```
{'context_precision': 1.0000, 'faithfulness': 1.0000, 'answer_relevancy': 0.9942, 'context_recall': 0.9667}
```
These metrics provide valuable insights into how well your chatbot's language model is performing based on the evaluation dataset.
