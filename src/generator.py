from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(name="gpt-4o-mini", temperature=0)

prompt = ChatPromptTemplate.from_template(
    """
        You are a precise assistant for customer support conversations.
        Answer the question using ONLY the information in the context below.

        Rules:
        - Use only information explicitly present in the context. Do not infer, assume, or add outside knowledge.
        - Answer directly. Do not restate or echo the question back in your answer.
        - Reproduce numbers, codes, names, and identifiers exactly as they appear in the context. Never approximate or summarize them.
        - Always answer in a complete sentence that includes the action or relationship the question implies. If asked what was "removed", say what it was AND that it was removed. If asked what "happened", describe the full event. Never answer with a noun phrase alone.
        - Stay strictly faithful to the roles described in the context. If the context says an agent is helping a customer, do not say the customer is helping. Do not reassign who did what.
        - If the context does not contain enough information to answer, say: "I do not have enough information in the provided context to answer that."
        - Keep the answer short, clear, and concise.

        Context:
        {context}

        Question: {question}

        Answer:
    """
)

chain = prompt | llm | StrOutputParser()

def generate(query: str, context: list[str]) -> str:
    """ Generates a grounded answer for the query and context chunks. """
    context_text = "\n\n".join(context)
    return chain.invoke({"question": query, "context": context_text})

if __name__ == "__main__":
    context = [
        "The customer purchased a new laptop last week but it is not working properly. Agent Annie asked for the order number, which was 123456789.",
        "The customer's monthly bill at Bright Star Services is usually around $80 but this month it came to $150. Agent Lisa reviewed the breakdown to find the source of the extra charges.",
        "Sophie's internet connection dropped while she was trying to send a work report. Her screen showed a no internet error with an exclamation point. She needed to send the report before 10 AM.",
        "Steven called Hotel California to book a room for 4 people on August 18th. Agent Candice took the reservation details.",
        "Customer Orlando Taylor noticed an unrecognized transaction on his debit account. The agent asked for the account number and verified his full name before pulling up the account information.",
    ]
    print(generate("Why did Steven called the hotel for?", context))