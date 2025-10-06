import getpass
import os
from langchain_groq import ChatGroq
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

def main():
    # Ask for API key if not already set
    while "GROQ_API_KEY" not in os.environ:
        os.environ["GROQ_API_KEY"] = getpass.getpass("enter your api key")
    else:
        print("api key initialized")
    initialize()


def initialize():
    # Create LLM
    llm = ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0,
        max_tokens=256,
        timeout=None,
        max_retries=2
    )

    # Reinstantiate memory every run
    memory = ConversationBufferMemory(return_messages=True)

    # Create conversation chain with memory
    conversation = ConversationChain(
        llm=llm,
        memory=memory,
        verbose=True
    )

    # Prepend system prompt as first input
    system_prompt = "You are a coding expert. Only output optimized code, no explanations, no language name, no intro, just code."
    print(conversation.predict(input=system_prompt))
    print(conversation.predict(input="give me fibonacci sequence function in python"))
    print(conversation.predict(input="optimize it for speed"))
    print(conversation.predict(input="optimize it for memory"))
    print(conversation.predict(input="make it iterative"))
    print(conversation.predict(input="add type hints"))
    print(conversation.predict(input="add docstring"))

if __name__ == "__main__":
    main()